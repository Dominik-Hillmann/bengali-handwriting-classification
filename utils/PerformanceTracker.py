class PerformanceTracker:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.epoch_train_losses = []
        self.epoch_train_acc = []
        self.epoch_val_losses = []
        self.epoch_val_acc = []

    
    def add_train(self, loss, acc):
        self.epoch_train_losses.append(loss)
        self.epoch_train_acc.append(acc)

    
    def add_val(self, loss, acc):
        self.epoch_val_losses.append(loss)
        self.epoch_val_acc.append(acc)


    def save(self, file_name):
        save_frame = pd.DataFrame({
            'losses_train': self.epoch_train_losses,
            'accuracies_train': self.epoch_train_acc,
            'losses_val': self.epoch_val_losses,
            'accuracies_val': self.epoch_val_acc
        })

        save_frame.to_csv(path.join(self.save_dir, file_name), quoting = csv.QUOTE_ALL)

    def graphs(self):
        
        epochs = list(range(1, self._get_num_epochs() + 1))
        # Losses        
        plt.plot(epochs, self.epoch_train_losses, 'r-', label = 'Training loss')
        plt.plot(epochs, self.epoch_val_losses, 'b-', label = 'Validation loss')
        plt.title('Average cross entropy loss by epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss metric')
        plt.legend(loc = 'upper right')
        plt.savefig(path.join(self.save_dir, 'losses.png'))

        plt.clf()

        # Accuracies
        plt.plot(epochs, self.epoch_train_acc, 'r-', label = 'Training accuracy')
        plt.plot(epochs, self.epoch_val_acc, 'b-', label = 'Validation accuracy')
        plt.title('Average accuracy by epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc = 'bottom right')
        plt.savefig(path.join(self.save_dir, 'accuracies.png'))

    

    def _get_num_epochs(self):
        num_train_loss = len(self.epoch_train_losses)
        num_train_acc = len(self.epoch_train_acc)
        num_val_losses = len(self.epoch_val_losses)
        num_val_acc = len(self.epoch_val_acc)
        same_lengths = (
            num_train_loss ==
            num_train_acc ==
            num_val_losses ==
            num_val_acc
        )

        if not same_lengths:
            raise ValueError('Not all metrics have the same number of observations.')
            
        return num_train_loss
