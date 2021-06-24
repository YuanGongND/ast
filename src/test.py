# -*- coding: utf-8 -*-
# @Time    : 6/21/21 4:51 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : test.py

class TransModelPretrainRaw10NewESC(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, pretrain=True):
        super(TransModelPretrainRaw10NewESC, self).__init__()

        # deit with pretraining
        if pretrain == True:
            print('now use pretrained deit')
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
            self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

            # automatcially get the intermediate shape
            test_img = torch.randn(1, 1, 128, 512)
            test_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(fstride, tstride))
            test_out = test_proj(test_img)
            f_dim = test_out.shape[2]
            t_dim = test_out.shape[3]
            num_patches = test_out.shape[2] * test_out.shape[3]
            print('number of patches:{:d}'.format(num_patches))

            # overlap time freq -10f-12t, modify the stride of patch embedding layer
            new_proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(fstride, tstride))
            print('f{:d}, t{:d} stride is used'.format(fstride, tstride))
            self.v.patch_embed.num_patches = num_patches
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # with transpose, input 128*1152, modify the position embedding
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 576, 768).transpose(1, 2).reshape(1, 768, 24, 24)
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(24, t_dim), mode='bilinear')
            print('now use bilinear interpolation')
            # new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(24, t_dim), mode='nearest')
            # print('now use nearest neighbor interpolation')
            new_pos_embed = new_pos_embed[:, :, 12 - int(f_dim/2): 12 - int(f_dim/2) + f_dim, :]
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1,2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            # print('now use reinitialized positional embedding')
            # new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches, 768))
            # trunc_normal_(new_pos_embed, std=.02)
            # self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:,:2,:].detach(), new_pos_embed], dim=1))

        # deit without pretraining
        else:
            print('now train from scratch')
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
            self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, label_dim))

            # automatcially get the intermediate shape
            test_img = torch.randn(1, 1, 128, 512)
            test_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(fstride, tstride))
            test_out = test_proj(test_img)
            f_dim = test_out.shape[2]
            t_dim = test_out.shape[3]
            num_patches = test_out.shape[2] * test_out.shape[3]
            print('number of patches:{:d}'.format(num_patches))

            # overlap time freq -10f-12t, modify the stride of patch embedding layer
            new_proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(fstride, tstride))
            print('f{:d}, t{:d} stride is used'.format(fstride, tstride))
            self.v.patch_embed.num_patches = num_patches
            # new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            # new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # random initialize the positional embedding
            new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, 768))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

    @autocast()
    def forward(self, x, nframe=1056):
        # # this only for baseline that has reverse order of time and feat dim
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.v.forward_features(x)
        x = (x[1] + x[0]) / 2
        x = self.mlp_head(x)
        return x