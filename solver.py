from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.r_dim = config.r_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_reg = config.lambda_reg
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.r_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.r_dim, self.d_repeat_num) 

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels):
        """Convert label indices to one-hot vectors.
        :param labels: N1
        """
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, self.c_dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_cls_labels(self, c_org):
        """Generate target domain labels for debugging and testing.
        :param c_org: N1
        """
        c_trg_list = []
        for i in range(self.c_dim):
            c_trg = self.label2onehot(torch.ones(c_org.size(0)) * i)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list
    
    def create_reg_labels(self, r_org):
        """Generate target va scores for debugging and testing.
        :param r_org: N2
        """
        centers = [
            [0.502170, 0.497995],
            [0.832602, 0.567080],
            [0.156761, 0.370705],
            [0.615316, 0.840581],
            [0.442205, 0.887155],
            [0.134421, 0.717098],
            [0.317687, 0.830228],
            [0.222583, 0.794049],
        ]
        r_trg_list = []
        for i in range(self.c_dim):
            r_trg = torch.tensor([centers[i]] * r_org.size(0))
            r_trg_list.append(r_trg.to(self.device))
        return r_trg_list

    def create_path_labels(self, batch_size, path):
        """Generate target cat and va values for testing.
        :param batch_size: batch size, may not equal to self.batch_size at last batch
        :param path: list of [cat, v, a]
        """
        c_trg_list = []
        r_trg_list = []
        for cat, v, a in path:
            c_trg = self.label2onehot(torch.tensor([cat] * batch_size))
            c_trg_list.append(c_trg.to(self.device))
            r_trg = torch.tensor([[v, a]] * batch_size)
            r_trg_list.append(r_trg.to(self.device))
        return c_trg_list, r_trg_list

    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def regression_loss(self, logit, target):
        """Compute mean squared error loss"""
        return F.mse_loss(logit, target)
    
    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org, r_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_cls_labels(c_org)
        r_fixed_list = self.create_reg_labels(r_org)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org, va_org = next(data_iter) # NCHW, N1, N2
            except:
                data_iter = iter(data_loader)
                x_real, label_org, va_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]      # N1
            va_trg = va_org[rand_idx]            # N2

            c_org = self.label2onehot(label_org) # N8
            c_trg = self.label2onehot(label_trg) # N8
            r_org = va_org.clone()               # N2
            r_trg = va_trg.clone()               # N2

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            r_org = r_org.to(self.device)             # Original va values.
            r_trg = r_trg.to(self.device)             # Target va values.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.
            va_org = va_org.to(self.device)           # VA values for computing regression loss.
            va_trg = va_trg.to(self.device)           # VA values for computing regression loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls, out_reg = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)
            d_loss_reg = self.regression_loss(out_reg, va_org)

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg, r_trg)
            out_src, out_cls, out_reg = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_reg * d_loss_reg + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_reg'] = d_loss_reg.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg, r_trg)
                out_src, out_cls, out_reg = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)
                g_loss_reg = self.regression_loss(out_reg, va_trg)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org, r_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_reg * g_loss_reg
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_reg'] = g_loss_reg.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed, r_fixed in zip(c_fixed_list, r_fixed_list):
                        x_fake_list.append(self.G(x_fixed, c_fixed, r_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_real, c_org, r_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_cls_labels(c_org)
                r_trg_list = self.create_reg_labels(r_org)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg, r_trg in zip(c_trg_list, r_trg_list):
                    x_fake_list.append(self.G(x_real, c_trg, r_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def testpath(self, path):
        """Translate images using StarGAN given (cat, v, a) path."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        data_loader = self.data_loader
        
        with torch.no_grad():
            for i, (x_real, _, _) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                batch_size = x_real.size(0)
                x_real = x_real.to(self.device)
                c_trg_list, r_trg_list = self.create_path_labels(batch_size, path)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg, r_trg in zip(c_trg_list, r_trg_list):
                    x_fake_list.append(self.G(x_real, c_trg, r_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))