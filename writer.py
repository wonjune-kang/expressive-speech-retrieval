from torch.utils.tensorboard import SummaryWriter


class MyWriter(SummaryWriter):
    def __init__(self, config, logdir):
        super(MyWriter, self).__init__(logdir)
        self.sample_rate = config.audio.sampling_rate

    def log_lr(self, lr, step):
        self.add_scalar("learning_rate", lr, step)

    def log_training(self, losses, step):
        for loss_type, value in losses.items():
            self.add_scalar(f"train/{loss_type}", value, step)

    def log_validation(self, losses, step):
        for loss_type, value in losses.items():
            self.add_scalar(f"validation/{loss_type}", value, step)

    def log_tsne_fig(self, fig, step):
        self.add_figure("val/tsne_embedding_viz", fig, step)
