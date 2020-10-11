import torch
from src.evaluation.evalloop import EvalLoop

class TrainLoop:
    def __init__(self, model, trainloader, optimizer, loss_fn, writer, valloader=None, print_every=100):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writer = writer
        self.print_every = print_every
        if valloader:
            self.eval = EvalLoop(model, valloader, writer)
        else:
            self.eval = EvalLoop(model, trainloader, writer)

    def fit(self, epochs=1):
        for epoch in range(epochs):
            running_loss = 0.0
            total_train = 0
            correct_train = 0
            for i, data in enumerate(self.trainloader):
                inputs, labels = data['x'], data['y']
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
                    print ("Loss: ", running_loss / self.print_every)
                    self.writer.add_scalar('training loss', running_loss / self.print_every, epoch * len(self.trainloader) + i)

                    # # ...log a Matplotlib Figure showing the model's predictions on a
                    # # random mini-batch
                    # writer.add_figure('predictions vs. actuals', plot_classes_preds(self.model, inputs, labels), global_step=epoch * len(trainloader) + i)
                    running_loss = 0.0

            train_accuracy = 100.0 * correct_train/total_train
            val_accuracy = self.eval.predict(epoch)
            print ("Epoch %d: Training Accuracy = %d%%,  Validation Accuracy = %d%%" % (epoch+1, train_accuracy, val_accuracy))

        print('Finished Training')