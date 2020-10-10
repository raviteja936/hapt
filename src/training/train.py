from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, model, trainloader, optimizer, loss_fn, writer):
        self.model = model
        self.trainloader = trainloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.writer = writer

    def fit(self, epochs=1):
        running_loss = 0.0
        for epoch in range(epochs):  # loop over the dataset multiple times
            for i, data in enumerate(self.trainloader):
                inputs, labels = data['x'], data['y']

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # every 1000 mini-batches...
                    # ...log the running loss
                    print ("Loss: ", running_loss / 100)
                    self.writer.add_scalar('training loss', running_loss / 100, epoch * len(self.trainloader) + i)
                    #
                    # # ...log a Matplotlib Figure showing the model's predictions on a
                    # # random mini-batch
                    writer.add_figure('predictions vs. actuals', plot_classes_preds(self.model, inputs, labels), global_step=epoch * len(trainloader) + i)
                    running_loss = 0.0
        print('Finished Training')