from features_v2 import feature_engineering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report , roc_auc_score
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader,TensorDataset
from pathlib import Path
import joblib


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def create_sequence(data, seq_len=12):
    return data.unfold(0, seq_len, 1).permute(0, 2, 1)


def spliting_data():

    (x_train,
    x_val1,
    x_val2,
    x_test,
    y_train,
    y_val1,
    y_val2,
    y_test
    ) = feature_engineering()
    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x_train.to_frame())
    x_val1_scaled = scaler.transform(x_val1.to_frame())
    x_val2_scaled = scaler.transform(x_val2.to_frame())
    x_test_scaled = scaler.transform(x_test.to_frame())

    file = Path(__file__).resolve().parent.parent / "model" / "scaler.pkl"
    joblib.dump(scaler,file)

#-----------------------------------------------------------------------
# Tensor conversion 
#-----------------------------------------------------------------------

    x_t = torch.tensor(x_scaled,dtype=torch.float32)
    x_val1_t = torch.tensor(x_val1_scaled,dtype=torch.float32)
    x_val2_t = torch.tensor(x_val2_scaled,dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled,dtype=torch.float32)

    y_t = torch.tensor(y_train.to_numpy(),dtype=torch.long)
    y_val1_t = torch.tensor(y_val1.to_numpy(),dtype=torch.long)
    y_val2_t = torch.tensor(y_val2.to_numpy(),dtype=torch.long)
    y_test_t = torch.tensor(y_test.to_numpy(),dtype=torch.long)

#-----------------------------------------------------------------------
# Creating chunks for training set
#-----------------------------------------------------------------------

    n = len(x_train)
    chunk1 = int(n * 0.25)
    chunk2 = int(n * 0.50)
    chunk3 = int(n * 0.75)

    x_chunk1 = x_t[:chunk1]
    x_chunk2 = x_t[chunk1:chunk2]
    x_chunk3 = x_t[chunk2:chunk3]
    x_chunk4 = x_t[chunk3:]

    y_chunk1 = y_t[:chunk1]
    y_chunk2 = y_t[chunk1:chunk2]
    y_chunk3 = y_t[chunk2:chunk3]
    y_chunk4 = y_t[chunk3:]

    SEQ_LEN = 12

#-----------------------------------------------------------------------
# Final split with 20 rows as 1 sequence  
#-----------------------------------------------------------------------

    x_chunk1 = create_sequence(x_chunk1)
    x_chunk2 = create_sequence(x_chunk2)
    x_chunk3 = create_sequence(x_chunk3)
    x_chunk4 = create_sequence(x_chunk4)

    x_val1_t = create_sequence(x_val1_t)
    x_val2_t = create_sequence(x_val2_t)
    x_test_t = create_sequence(x_test_t)

    y_chunk1 = y_chunk1[SEQ_LEN - 1:]
    y_chunk2 = y_chunk2[SEQ_LEN - 1:]
    y_chunk3 = y_chunk3[SEQ_LEN - 1:]
    y_chunk4 = y_chunk4[SEQ_LEN - 1:]

    y_val1_t = y_val1_t[SEQ_LEN - 1:]
    y_val2_t = y_val2_t[SEQ_LEN - 1:]
    y_test_t = y_test_t[SEQ_LEN - 1:]

    return x_chunk1,x_chunk2,x_chunk3,x_chunk4,x_val1_t,x_val2_t,x_test_t,y_chunk1,y_chunk2,y_chunk3,y_chunk4,y_val1_t,y_val2_t,y_test_t


class Transformer(nn.Module):
    def __init__(
        self,
        n_features=1,
        seq_len=12,
        d_model=32,
        n_head=1,
        n_layers=2,
        dim_foreforward=64,
        dim_output=2,
        dropout=0.1
    ):
        super().__init__()

        # Input embedding
        self.embed = nn.Linear(n_features, d_model)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder block
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_foreforward,
            dropout=dropout,
            batch_first=True
        )

        self.encod = nn.TransformerEncoder(
            self.block,
            num_layers=n_layers
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_output)
        )

    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos_embed
        x = self.encod(x)
        x = x[:, -1, :]
        return self.head(x)



def evaluate_model(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    predictions = []
    probabilities = []
    labels = []
    losses = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            probs = torch.softmax(output, dim=1)[:, 1]
            preds = torch.argmax(output, dim=1)
            loss = loss_fn(output, y_batch)

            predictions.append(preds.cpu())
            probabilities.append(probs.cpu())
            labels.append(y_batch.cpu())
            losses.append(loss)

    predictions = torch.cat(predictions)
    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels)
    loss = torch.stack(losses).mean()

    accuracy = (predictions == labels).float().mean()
    return {
        "accuracy": accuracy.item(),
        "loss": loss.item(),
        "auc": roc_auc_score(labels.numpy(), probabilities.numpy()),
        "report": classification_report(labels.numpy(), predictions.numpy()),
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": labels,
    }


def main():
    global device, x_train, x_val1, x_val2, x_test, y_train, y_val1, y_val2, y_test

    (x_train,x_val1,x_val2,x_test,y_train,y_val1,y_val2,y_test) = feature_engineering()
    
    (
    x_chunk1,x_chunk2,x_chunk3,x_chunk4,x_val1_t,x_val2_t,x_test_t,
    y_chunk1,y_chunk2,y_chunk3,y_chunk4,y_val1_t,y_val2_t,y_test_t
    ) = spliting_data()
    print("Tensors fromed ................ ")
    
    validation_1_dataset = TensorDataset(
        x_val1_t,
        y_val1_t,
    )
    
    validation_1_loader = DataLoader(
        dataset=validation_1_dataset,
        batch_size=256,
        shuffle=False,
    )
    print("Batches Created ................ ")
    
    model = Transformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    CHUNK_TO_TRAIN = 4
    CHUNK_EPOCHS = 20
    CHECKPOINT_PATH = "transformer_checkpoint.pth"
    
    
    chunks = {
        1: ("chunk_1", x_chunk1, y_chunk1),
        2: ("chunk_2", x_chunk2, y_chunk2),
        3: ("chunk_3", x_chunk3, y_chunk3),
        4: ("chunk_4", x_chunk4, y_chunk4),
    }
    
    project = Path(__file__).resolve().parent.parent
    data = project / "model" / CHECKPOINT_PATH
    
    if CHUNK_TO_TRAIN > 1:
        checkpoint = torch.load(data, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    chunk_name, x_chunk, y_chunk = chunks[CHUNK_TO_TRAIN]
    
    chunk_loader = DataLoader(
        dataset=TensorDataset(x_chunk, y_chunk),
        batch_size=256,
        shuffle=False,
    )
    
    for epoch in range(CHUNK_EPOCHS):
        model.train()
        chunk_loss = 0.0

        for x_batch, y_batch in chunk_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            chunk_loss += loss.item()

        print(
            f"{chunk_name} | Epoch {epoch + 1}/{CHUNK_EPOCHS} | "
            f"Train loss: {chunk_loss / len(chunk_loader):.4f}",
            flush=True,
        )
    
    torch.save(
        {
            "completed_chunk": CHUNK_TO_TRAIN,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        data,
    )

    val1_metrics = evaluate_model(model, validation_1_loader)
    print(
        f"{chunk_name} validation | "
        f"Accuracy: {val1_metrics['accuracy']:.4f} | "
        f"Loss: {val1_metrics['loss']:.4f} | "
        f"AUC: {val1_metrics['auc']:.4f}",
        flush=True,
    )
    print(f"Classification report:\n{val1_metrics['report']}")

if __name__ == "__main__":
    main()