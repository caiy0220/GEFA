import torch
from gefa import utils
from gefa.utils import log
# from gefa import text_utils
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

class GEFA4Image(object):
    def __init__(self, sample_num, input_size, baseline=None, filter_w=31, filter_s=5, grid_num=11):
        # Input size: tuple specifying (C, H, W)
        self.verbose = False
        self.sample_num = sample_num // 2 
        self.grid_num, self.fix_grids = grid_num, True

        self.input_size = torch.tensor(input_size)
        self.baseline = torch.zeros(input_size) if baseline is None else baseline
        self.smoother = None if filter_w <= 1 else GaussianBlur(filter_w, filter_s).cuda() 

        log('GEFA initializing query masks, might take some time')
        masks, paths = self._get_init_masks()   # Get masks on path
        self.masks, self.paths = self._mask_path_overlay(masks, paths)
        self.mask_weights = self._get_mask_weights()    # Get mask weights depending on gamma
        self.blur_m = GaussianBlur((31, 31), 5.0)
        self.flag_cv = True     # Enable control variate by default

    def _get_init_masks(self):
        # Do oversampling to exclude repeated sampling of full absence/presence 
        oversampling = int(self.sample_num + 2 * (self.sample_num / (self.input_size[-2:].prod()) + 1))
        masks = self._get_masks(oversampling)
        paths = self._get_paths(oversampling)
        return masks, paths
    
    def explain(self, m, x, batch_size=64, verbose=False, target_class=None, visual_enhanced=False):
        total_n, baseline = len(self.masks), self._get_baseline(x)  # TBD: check the baseline
        device = utils.get_device(m)
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.to(device)

        with torch.no_grad():
            x = x.to(device)
            fss = []
            lbl = m(x).cpu().argmax().item()    
            # Use the most activated class as the target unless explicitly specified
            target_class = target_class if target_class else lbl
            for ptr_l in tqdm(range(0, total_n, batch_size), disable=not verbose):
                ptr_r = ptr_l+batch_size
                b_masks = self.masks[ptr_l:ptr_r].to(device)
                # Query generation with pre-constructed masks
                queries = self._get_queries(x, baseline, b_masks)   
                fss.append(self._fitness(m, queries, target_class).cpu())

            fss = torch.concatenate(fss)
            ges = self._gradient_estimate(fss)
            attr = torch.mean(ges, dim=0) 

            diff = (x - baseline)[0].cpu().abs()
            enhanced = attr*diff
            return (attr, enhanced) if visual_enhanced else attr
        
    def _get_queries(self, x, baseline, masks):
        return torch.add(masks * x, ~masks * baseline)
        
    def _gradient_estimate(self, fitness):
        hs = self._get_control_variate(self.masks)
        beta = self._estimate_beta(fitness, hs) 
        fitness = fitness - hs * beta

        fitness = fitness.reshape((-1, 1, 1, 1))
        return (fitness.cuda() * self.mask_weights.cuda()).cpu()
    
    def _fitness(self, m, imgs, target=None):
        device = utils.get_device(m)
        logits = m(imgs.to(device))
        return logits[:, target]
    
    def _get_padded_size(self):
        if self.smoother is None:
            return self.input_size[-2], self.input_size[-2]
        extended_h = self.input_size[-2] + self.smoother.kernel_size[0] - 1
        extended_w = self.input_size[-1] + self.smoother.kernel_size[1] - 1
        return extended_h, extended_w

    def _get_crop_idx(self, h, w):
        if self.smoother is None:
            return (0, h), (0, w)
        half_kernel_h = (self.smoother.kernel_size[0]-1) // 2
        half_kernel_w = (self.smoother.kernel_size[1]-1) // 2
        h_ptr = (half_kernel_h, h - half_kernel_h)
        w_ptr = (half_kernel_w, w - half_kernel_w)
        return h_ptr, w_ptr

    def _get_masks(self, num_oversampling):
        num_features = self.input_size[-2:].prod()
        # ----- Apply mask smoothing for dimension reduction -----
        extended_h, extended_w = self._get_padded_size()
        masks = torch.rand((num_oversampling, 1, extended_h, extended_w))
        h_ptr, w_ptr = self._get_crop_idx(extended_h, extended_w)
        masks = self._mask_smoothing(masks)[:, :, h_ptr[0]:h_ptr[1], w_ptr[0]:w_ptr[1]].flatten()
        masks = self._reassign_mask_values(masks, num_features, self.verbose)
        return masks.reshape((num_oversampling, 1, *self.input_size[-2:].tolist()))

    def _get_paths(self, num_oversampling):
        range_l, range_r = 0., 1.
        if self.fix_grids:
            grids = torch.linspace(range_l, range_r, self.grid_num+2)[1:-1]
        else:
            grids = torch.linspace(range_l, range_r, self.grid_num+1)[:-1]
            width = grids[1] - grids[0]
        
        repeats = num_oversampling // self.grid_num
        residual = num_oversampling % self.grid_num
        residual_idxs = torch.randperm(self.grid_num)[:residual]
        residual_grids = grids[residual_idxs]

        grids = torch.concat([grids.tile(repeats), residual_grids])

        if self.fix_grids:
            return grids.reshape(-1,1,1,1)

        grid_samples = torch.rand_like(grids) * width
        return (grids + grid_samples).reshape(-1,1,1,1)
 
    def _mask_smoothing(self, masks, batch_size=64):
        if self.smoother is None:
            return masks
        buff = []
        with torch.no_grad():
            for ptr_l in range(0, len(masks), batch_size):
                ptr_r = ptr_l + batch_size
                batch = masks[ptr_l:ptr_r]
                batch = self.smoother(batch.cuda()).cpu()   # Smoothing through low-pass
                buff.append(batch)
        return torch.concat(buff)
    
    def _reassign_mask_values(self, masks, batch_size, verbose=False):
        if self.smoother is None:
            return masks
        buff = []
        with torch.no_grad():
            for ptr_l in tqdm(range(0, len(masks), batch_size), disable=not verbose):
                ptr_r = ptr_l + batch_size
                batch = masks[ptr_l:ptr_r]
                batch_ranking = torch.sort(batch.cuda()).indices.cpu()
                pseudo_uniforms = get_ranked_pseudo_uniform(len(batch))

                batch_uniform = torch.zeros_like(pseudo_uniforms)
                batch_uniform[batch_ranking] = pseudo_uniforms
                buff.append(batch_uniform)
        return torch.concat(buff)
    
    def _mask_path_overlay(self, masks, paths):
        hits = masks < paths
        presence_counts = hits.sum(dim=(1,2,3)) 

        num_features = self.input_size[-2:].prod()
        not_full_abs = set(torch.where(presence_counts != 0)[0].tolist())
        not_full_prs = set(torch.where(presence_counts != num_features)[0].tolist())
        valid_idxs = torch.tensor(list(not_full_abs & not_full_prs))
        hits, paths = hits[valid_idxs][:self.sample_num], paths[valid_idxs][:self.sample_num]

        # Antithetic Sampling
        hits = torch.concat([hits, ~hits]) 
        paths = torch.concat([paths, 1-paths]) 
        return hits, paths
    
    def _get_mask_weights(self):
        alphas, c_alphas = self.paths, 1-self.paths
        masks_in_float = self.masks.to(torch.float)
        ws = masks_in_float / alphas + (masks_in_float-1) / c_alphas
        return ws
    
    def _get_control_variate(self, masks):
        num_features = torch.tensor(masks.shape[-2:]).prod()
        hs = masks.sum(dim=(1,2,3)) / num_features - 0.5
        hs[hs == 1.] = 0.
        return hs
    
    def _estimate_beta(self, ys, hs):
        obs = torch.cat([ys.unsqueeze(0), hs.unsqueeze(0)])
        cov_mat = torch.cov(obs)
        cov, var_h = cov_mat[0, 1], cov_mat[1, 1]
        return cov/var_h
    
    def _get_baseline(self, x):
        if isinstance(self.baseline, str) and self.baseline.lower() == 'blur':   
            # explicand-specific baseline, requiring definition of 'self.blur_m', only apply to IMG
            if self.blur_m is None: 
                print('Bluring kernel is not defined, use default')
                self.set_bluring_kernel()
            baselines = self.blur_m(x)
        elif isinstance(self.baseline, (int, float)):
            # Constant value baseline, identical value across different channels (if applies)
            baselines = self.baseline
        elif isinstance(self.baseline, torch.Tensor):
            # Image-like baseline, having shape [C*H*W]
            baselines = self.baseline
        elif isinstance(self.baseline, tuple):
            # Constant value baseline for multi-channel image only
            baselines = torch.Tensor(self.baseline).reshape(len(self.baseline),1,1)
        else:
            assert 1 == 0, 'Unsupported baseline type'
        return baselines

    def set_bluring_kernel(self, blur_m=None):
        """
        Parameters:
        -----------
        blur_m: blurring kernel for acquiring the baseline, 
                used only when specifying self.baseline='blur'
        """
        self.blur_m = blur_m if blur_m is not None else GaussianBlur((31, 31), 5.0)
    
def get_ranked_pseudo_uniform(num):
    grids = torch.linspace(0, 1, num)
    pseudo_uniforms = grids + torch.rand_like(grids) / num
    return pseudo_uniforms
