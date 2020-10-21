import torch
from src.evaluation.evalloop import EvalLoop

class TrainLoop:
    def __init__(self, model, train_loader, optimizer, loss_fn, device, writer, val_loader=None, print_every=100):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.writer = writer
        self.print_every = print_every
        if val_loader:
            self.eval = EvalLoop(model, val_loader, self.device)
        else:
            self.eval = EvalLoop(model, train_loader, self.device)

    def fit(self, epochs=1):
        for epoch in range(epochs):
            running_loss = 0.0
            total_train = 0
            correct_train = 0

            for i, data in enumerate(self.train_loader):
                inputs, labels = data['x'], data['y']
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total_train += inputs.shape[0]
                correct_train += predicted.eq(labels.data).sum().item()
                running_loss += loss.item()

                if i % self.print_every == self.print_every-1:
                    print("Loss: ", running_loss / self.print_every)
                    self.writer.add_scalar("Training loss", running_loss / self.print_every, 10 * epoch + int((i * 10) / len(self.train_loader)))
                    running_loss = 0.0

            train_accuracy = 100.0 * correct_train/total_train
            val_accuracy, val_cm = self.eval.predict()
            self.writer.add_scalar("Training accuracy", train_accuracy, epoch)
            self.writer.add_scalar("Validation accuracy", val_accuracy, epoch)
            print("Epoch %d: Training Accuracy = %d%%,  Validation Accuracy = %d%%" % (epoch+1, train_accuracy, val_accuracy))
            # print(val_cm)
        print("Finished Training")