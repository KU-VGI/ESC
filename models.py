from torch import nn


def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Parameter Initialization
def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)

        self.embed_dim = n_filter2

        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )

        # for consistency with other models (ViT)
        self.head = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

        # for SVD method
        self.singular_vectors = None


    def forward(self, x, all=False):
        features = self.features(x)

        # ESC
        if hasattr(self, 'esc'):
            features = (self.esc @ self.esc.T @ features.T).T

        output = self.head(features)

        if all:
            res = dict()
            res['pre_logits'] = features
            res['logits'] = output

            return res

        return output

    def esc_set(self, u, esc_t=False):
        if esc_t:
            if hasattr(self, 'esc'):
                self.esc = u
            else:
                self.register_buffer('esc', u.T)
        else:
            if hasattr(self, 'esc'):
                self.esc = u @ self.esc
            else:
                self.register_buffer('esc', u)



from timm.models.registry import register_model
from timm.models import create_model

from vision_transformer import _create_vision_transformer

__all__ = [
    'vit_base_patch16_224', 'vit_base_patch16_384',
]

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


def load_vit(model_name, num_classes=10, device='cpu', is_pretrained=True, is_backbone_freezed=True):

    print(f"Creating model: {model_name}")
    model = create_model(
        model_name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
    )
    model.to(device)  
    
    freeze = ['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed']
    
    if is_backbone_freezed:
        for n, p in model.named_parameters():
            if n.startswith(tuple(freeze)):
                p.requires_grad = False
    
    return model
