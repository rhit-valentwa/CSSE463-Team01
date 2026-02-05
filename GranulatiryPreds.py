import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import segmentation, color
from scipy import ndimage


# ============================================================================
# TECHNIQUE 1: Edge-Aware Loss Function
# ============================================================================

class EdgeAwareLoss(nn.Module):
    """
    Loss function that penalizes color bleeding across edges.
    Uses Sobel filters to detect edges in the L channel and applies
    higher weight to predictions near boundaries.
    """
    def __init__(self, edge_weight=2.0, base_loss='l1'):
        super(EdgeAwareLoss, self).__init__()
        self.edge_weight = edge_weight
        self.base_loss = base_loss
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def detect_edges(self, L_channel):
        """
        Detect edges in the luminance channel using Sobel filters.
        
        Args:
            L_channel: (B, 1, H, W) tensor of L channel values
            
        Returns:
            edge_map: (B, 1, H, W) tensor with higher values at edges
        """
        # Apply Sobel filters
        edge_x = F.conv2d(L_channel, self.sobel_x, padding=1)
        edge_y = F.conv2d(L_channel, self.sobel_y, padding=1)
        
        # Compute edge magnitude
        edge_map = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        
        # Normalize to [0, 1] range
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-6)
        
        return edge_map
    
    def forward(self, pred_ab, target_ab, L_channel):
        """
        Compute edge-aware loss.
        
        Args:
            pred_ab: (B, 2, H, W) predicted a* and b* channels
            target_ab: (B, 2, H, W) ground truth a* and b* channels
            L_channel: (B, 1, H, W) input L channel
            
        Returns:
            loss: scalar tensor
        """
        # Compute base loss (L1 or L2)
        if self.base_loss == 'l1':
            pixel_loss = torch.abs(pred_ab - target_ab)
        else:  # l2
            pixel_loss = (pred_ab - target_ab)**2
        
        # Detect edges
        edge_map = self.detect_edges(L_channel)
        
        # Create edge weight map: higher weight at edges
        weight_map = 1.0 + self.edge_weight * edge_map
        
        # Apply weights to loss
        weighted_loss = pixel_loss * weight_map
        
        return weighted_loss.mean()


# ============================================================================
#  Guided Filter for Edge-Preserving Smoothing
# ============================================================================

class GuidedFilter(nn.Module):
    """
    Guided filter that uses the L channel as guidance to preserve edges
    while smoothing color predictions within regions.
    """
    def __init__(self, radius=5, eps=0.01):
        super(GuidedFilter, self).__init__()
        self.radius = radius
        self.eps = eps
    
    def box_filter(self, x, r):
        """Fast box filter using cumulative sum."""
        batch, ch, h, w = x.shape
        
        # Pad the input
        x_pad = F.pad(x, (r, r, r, r), mode='reflect')
        
        # Use average pooling for box filter
        kernel_size = 2 * r + 1
        box = F.avg_pool2d(x_pad, kernel_size, stride=1, padding=0) * (kernel_size ** 2)
        
        return box
    
    def forward(self, guidance, input_tensor):
        """
        Apply guided filter.
        
        Args:
            guidance: (B, 1, H, W) guidance image (L channel)
            input_tensor: (B, C, H, W) input to be filtered (predicted ab channels)
            
        Returns:
            output: (B, C, H, W) filtered output
        """
        r = self.radius
        
        # Mean of guidance
        mean_I = self.box_filter(guidance, r) / ((2*r+1)**2)
        
        # Mean of input
        mean_p = self.box_filter(input_tensor, r) / ((2*r+1)**2)
        
        # Correlation of I and p
        corr_Ip = self.box_filter(guidance * input_tensor, r) / ((2*r+1)**2)
        
        # Covariance of I and p
        cov_Ip = corr_Ip - mean_I * mean_p
        
        # Variance of guidance
        var_I = self.box_filter(guidance * guidance, r) / ((2*r+1)**2) - mean_I * mean_I
        
        # Compute a and b
        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        
        # Mean of a and b
        mean_a = self.box_filter(a, r) / ((2*r+1)**2)
        mean_b = self.box_filter(b, r) / ((2*r+1)**2)
        
        # Output
        output = mean_a * guidance + mean_b
        
        return output


# ============================================================================
# Superpixel-Based Post-Processing
# ============================================================================

def superpixel_refinement(L_channel, ab_pred, n_segments=500, compactness=10):
    """
    Refine color predictions using superpixel segmentation.
    Colors within each superpixel are averaged, reducing color bleeding.
    
    Args:
        L_channel: (H, W) numpy array of L channel
        ab_pred: (H, W, 2) numpy array of predicted a* and b* channels
        n_segments: number of superpixels
        compactness: controls superpixel shape (higher = more compact)
        
    Returns:
        refined_ab: (H, W, 2) refined a* and b* predictions
    """
    # Convert L channel to 0-255 range for SLIC
    L_normalized = ((L_channel - L_channel.min()) / 
                   (L_channel.max() - L_channel.min()) * 255).astype(np.uint8)
    
    # Generate superpixels using SLIC
    segments = segmentation.slic(L_normalized, n_segments=n_segments, 
                                 compactness=compactness, start_label=0)
    
    # Initialize refined output
    refined_ab = np.zeros_like(ab_pred)
    
    # Average colors within each superpixel
    for segment_id in np.unique(segments):
        mask = segments == segment_id
        
        # Compute mean color for this superpixel
        mean_a = ab_pred[mask, 0].mean()
        mean_b = ab_pred[mask, 1].mean()
        
        # Assign mean color to all pixels in this superpixel
        refined_ab[mask, 0] = mean_a
        refined_ab[mask, 1] = mean_b
    
    return refined_ab


# ============================================================================
# Bilateral Filter for Edge-Preserving Smoothing
# ============================================================================

def bilateral_filter_ab(L_channel, ab_pred, sigma_spatial=5, sigma_color=0.1):
    """
    Apply bilateral filter to predicted ab channels using L channel as guidance.
    Smooths colors within regions while preserving edges.
    
    Args:
        L_channel: (H, W) numpy array of L channel (normalized 0-1)
        ab_pred: (H, W, 2) numpy array of predicted a* and b* channels
        sigma_spatial: spatial smoothness parameter
        sigma_color: color/range parameter
        
    Returns:
        filtered_ab: (H, W, 2) filtered a* and b* predictions
    """
    from skimage.restoration import denoise_bilateral
    
    # Combine L and ab for joint bilateral filtering
    Lab_pred = np.dstack([L_channel, ab_pred])
    
    # Apply bilateral filter
    filtered_Lab = denoise_bilateral(
        Lab_pred,
        sigma_spatial=sigma_spatial,
        sigma_color=sigma_color,
        channel_axis=-1
    )
    
    # Extract filtered ab channels
    filtered_ab = filtered_Lab[:, :, 1:]
    
    return filtered_ab


# ============================================================================
# Multi-Scale Feature Aggregation Module
# ============================================================================

class MultiScaleBoundaryModule(nn.Module):
    """
    Neural network module that aggregates features at multiple scales
    to better capture object boundaries at different granularities.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBoundaryModule, self).__init__()
        
        # Different dilation rates to capture multi-scale context
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=8, dilation=8)
        
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Extract features at different scales
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        # Fuse features
        output = self.fusion(multi_scale)
        output = self.bn(output)
        output = self.relu(output)
        
        return output


# ============================================================================
# Attention-Based Boundary Refinement
# ============================================================================

class BoundaryAttentionModule(nn.Module):
    """
    Attention module that focuses on boundary regions to improve
    color predictions at object edges.
    """
    def __init__(self, in_channels):
        super(BoundaryAttentionModule, self).__init__()
        
        # Boundary detection branch
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Feature refinement
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        # Generate boundary attention map
        attention = self.boundary_conv(features)
        
        # Refine features
        refined = self.refine_conv(features)
        
        # Apply attention (emphasize boundary regions)
        output = features + attention * refined
        
        return output


class ImprovedColorizationModel(nn.Module):
    """
    Example of integrating boundary improvement techniques into your model.
    This adds the multi-scale and attention modules to your existing architecture.
    """
    def __init__(self, base_model):
        super(ImprovedColorizationModel, self).__init__()
        
        self.base_model = base_model
        
        # Add boundary refinement modules
        # Assuming your model outputs 64 channels before final conv
        self.multi_scale = MultiScaleBoundaryModule(64, 64)
        self.boundary_attention = BoundaryAttentionModule(64)
        
        # Final convolution to ab channels
        self.final_conv = nn.Conv2d(64, 2, 1)
        self.tanh = nn.Tanh()
    
    def forward(self, L_channel):
        # Extract features from base model (modify to match your architecture)
        features = self.base_model.extract_features(L_channel)
        
        # Apply multi-scale aggregation
        features = self.multi_scale(features)
        
        # Apply boundary attention
        features = self.boundary_attention(features)
        
        # Generate final ab predictions
        ab_pred = self.final_conv(features)
        ab_pred = self.tanh(ab_pred)
        
        return ab_pred


# ============================================================================
# TRAINING SCRIPT WITH EDGE-AWARE LOSS
# ============================================================================

def train_with_improved_boundaries(model, train_loader, val_loader, 
                                   epochs=10, lr=0.0005, device='cuda'):
    """
    Training script using edge-aware loss for better boundaries.
    """
    model = model.to(device)
    
    # Use edge-aware loss
    criterion = EdgeAwareLoss(edge_weight=2.0, base_loss='l1')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Optional: Guided filter for post-processing
    guided_filter = GuidedFilter(radius=5, eps=0.01)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (L_channel, ab_target) in enumerate(train_loader):
            L_channel = L_channel.to(device)
            ab_target = ab_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            ab_pred = model(L_channel)
            
            # Compute edge-aware loss
            loss = criterion(ab_pred, ab_target, L_channel)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for L_channel, ab_target in val_loader:
                L_channel = L_channel.to(device)
                ab_target = ab_target.to(device)
                
                ab_pred = model(L_channel)
                
                # Optional: Apply guided filter during validation
                ab_pred = guided_filter(L_channel, ab_pred)
                
                loss = criterion(ab_pred, ab_target, L_channel)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss/len(train_loader):.6f}")
        print(f"  Val Loss: {val_loss/len(val_loader):.6f}")


# ============================================================================
# POST-PROCESSING PIPELINE
# ============================================================================

def apply_boundary_refinement_pipeline(L_channel_np, ab_pred_np, 
                                      use_superpixel=True,
                                      use_bilateral=True):
    """
    Complete post-processing pipeline to refine color boundaries.
    
    Args:
        L_channel_np: (H, W) numpy array
        ab_pred_np: (H, W, 2) numpy array
        use_superpixel: whether to apply superpixel refinement
        use_bilateral: whether to apply bilateral filtering
        
    Returns:
        refined_ab: (H, W, 2) refined predictions
    """
    refined_ab = ab_pred_np.copy()
    
    # Step 1: Superpixel-based refinement
    if use_superpixel:
        print("Applying superpixel refinement...")
        refined_ab = superpixel_refinement(L_channel_np, refined_ab, 
                                          n_segments=500, compactness=10)
    
    # Step 2: Bilateral filtering
    if use_bilateral:
        print("Applying bilateral filter...")
        refined_ab = bilateral_filter_ab(L_channel_np, refined_ab,
                                        sigma_spatial=5, sigma_color=0.1)
    
    return refined_ab


# ============================================================================
# COMPLETE EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    
    # Example 1: Add edge-aware loss to training
    print("=" * 80)
    print("TECHNIQUE 1: Edge-Aware Loss")
    print("=" * 80)
    
    
    # Example 2: Apply post-processing to predictions
    print("\n" + "=" * 80)
    print("TECHNIQUE 2: Post-Processing Pipeline")
    print("=" * 80)
    
    
    # Example 3: Modify model architecture
    print("\n" + "=" * 80)
    print("TECHNIQUE 3: Enhanced Model Architecture")
    print("=" * 80)
  
    