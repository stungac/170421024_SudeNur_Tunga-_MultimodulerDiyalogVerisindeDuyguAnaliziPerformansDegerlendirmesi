import torch.nn as nn
import torch
import pickle
import numpy as np

# 1024 -> 600 dönüşüm katmanı
class DimensionalityReduction(nn.Module):
    def __init__(self):
        super(DimensionalityReduction, self).__init__()
        self.linear = nn.Linear(1024, 600)

    def forward(self, x):
        x = self.linear(x)
        return x
    
# model oluşturuldu
reduction_model = DimensionalityReduction()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# reduction_model.to(device)
reduction_model.eval()

_, _, _, dict_input, _, _, _, _, _, _, _ = pickle.load(open('meld_features_roberta.pkl', 'rb'))

# sonuçları tutacak yeni dictionary oluşturuldu
new_videoText = {}

# liste numpy arrayinei ardından da pytorch tensörüne çevrildi. Modelden geçirilerek boyut düşürüldü
for key, value in dict_input.items():
    value = np.array(list(value))
    tensor = torch.tensor(value, dtype=torch.float32)
    with torch.no_grad():  # gradient hesaplanmasın
        output = reduction_model(tensor)  # (cümle sayısı, 600)
    new_videoText[key] = output.numpy()

# yeni dictionary pickle dosyasına kaydedildi
with open('meld_features_roberta_600.pkl', 'wb') as f:
    pickle.dump(new_videoText, f)
