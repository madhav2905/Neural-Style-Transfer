from dataset import StyleTransferDataset
from torch.utils.data import DataLoader

dataset = StyleTransferDataset("data/content", "data/style", image_size=256)

print("Dataset length:", len(dataset))

content, style = dataset[0]
print("Content shape:", content.shape)
print("Style shape:", style.shape)
print("Content value range:", content.min().item(), content.max().item())

loader = DataLoader(dataset, batch_size=4, shuffle=True)
content_batch, style_batch = next(iter(loader))
print("Batch content shape:", content_batch.shape)
print("Batch style shape:", style_batch.shape)