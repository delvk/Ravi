import pandas as pd 
if __name__ == "__main__":
    link = "log/Keras_MCNN_training_bak.log"
    data = pd.read_csv(link)
    min_val = min(data["val_mae"])
    e = -1
    for i in range(len(data)):
        if data["val_mae"][i] == min_val:
            e = data["epoch"][i]
    print(e)