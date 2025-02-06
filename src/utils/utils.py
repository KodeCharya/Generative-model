def save_model(model, filepath):
    import torch
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    import torch
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode
    return model