import matplotlib.pyplot as plt
import json

for file in ["deez_nuts_bert.json", "history.json"]:
    data = json.load(open(file))
    plt.plot(data['loss'], label='loss')
    plt.plot(data['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.show()