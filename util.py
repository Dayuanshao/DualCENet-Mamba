import torch

# 加载 .pth 文件
checkpoint = torch.load("/root/autodl-tmp/weifuyuan/projects/DuleCENet-master/weights/LOLv1/w_perc.pth", map_location="cpu")

def replace_lca_with_cseb(obj):
    if isinstance(obj, dict):
        return {replace_lca_with_cseb(k): replace_lca_with_cseb(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_lca_with_cseb(item) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("LCA", "CSEB")
    else:
        return obj

# 替换内容
new_checkpoint = replace_lca_with_cseb(checkpoint)

# 保存到新文件
torch.save(new_checkpoint, "epoch_770_cseb.pth")

print("完成替换并保存为 epoch_770_cseb.pth")
