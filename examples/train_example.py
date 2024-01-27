from absl import flags
import torch
from torch import nn, optim
import tqdm
from torch.utils.data import DataLoader
from rtx.data.dataset import get_oxe_dataset, TorchRLDSDataset
from tensorboardX import SummaryWriter
import os
import tensorflow as tf
import pytorch_warmup as warmup
tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_string("dataset_name", "fractal20220817_data", "Dataset name.")
flags.DEFINE_string("eval_against", "", "Model to eval against in format '<model_key>/<weights_key>. See robo-transformers pypi for options.")



def run(model: torch.nn.Module, action_tokenizer):
    writer = SummaryWriter()
    train_ds = TorchRLDSDataset(get_oxe_dataset(FLAGS.dataset_name, train=True))
    eval_ds = TorchRLDSDataset(get_oxe_dataset(FLAGS.dataset_name, train=False))
 
    train_data_loader = DataLoader(
        train_ds,
        batch_size=FLAGS.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    eval_data_loader = DataLoader(
        eval_ds,
        batch_size=FLAGS.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    steps_per_epoch = 5900
    warmup_period = 1000
    num_steps = steps_per_epoch * FLAGS.num_epochs - warmup_period
    t0 = num_steps // 15
    lr_min = 1e-5
    max_step = t0 * 3 + warmup_period

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=2, eta_min=lr_min)

    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

    step_num = 0
    for epoch in range(FLAGS.num_epochs):
        print(f'epoch {epoch}')
        for i, sample in tqdm.tqdm(enumerate(train_data_loader)):
            # batch, frames, height, width, channels -> batch, channels, frames, height, width
            with torch.no_grad():
                video = (torch.permute(sample['observation']['image_primary'],(0,1,4,2,3)) / 255.0).to(device)
                instructions = sample['language_instruction']
                ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action']).reshape(-1,1).squeeze().long().to(device)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            out = model.train_step(video, instructions).reshape(-1, 256)
            loss = criterion(out, ground_truth)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.to('cpu').detach().numpy(), step_num)
            del video
            del instructions
            del ground_truth
            # if (i+1) % 10 == 0:
            #     # writer.add_image('last img first sample', sample['observation']['image_primary'][0,-1,:,:,:].numpy(), step_num)
            #     write_dict_to('last_action_first_batch_sample', writer, {'act': sample['action'][0,-1,:]} , step_num)
            #     writer.add_text('instruction_first_batch_sample', sample['language_instruction'][0], step_num)
            step_num += 1

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
            if warmup_scheduler.last_step + 1 >= max_step:
                break
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step_num)
        
        # evaluate
        print('evaluating')

        model.eval()
        with torch.no_grad():
            eval_loss = 0
            single_eval_loss = 0
            eval_steps = 0.
            for i, sample in tqdm.tqdm(enumerate(eval_data_loader)):
                eval_steps += 1
                video = (torch.permute(sample['observation']['image_primary'],(0,1,4,2,3)) / 255.0).to(device)
                instructions = sample['language_instruction']
                ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action']).long().to(device)
                out = model.run(video, instructions)

                eval_loss += criterion(out.reshape(-1, 256), ground_truth.reshape(-1,1).squeeze()).to('cpu')
                single_eval_loss += criterion(out[:,-1,:,:].reshape(-1, 256), ground_truth[:,-1,:].reshape(-1,1).squeeze()).to('cpu')

        writer.add_scalar('eval_loss', eval_loss / eval_steps, step_num)
        writer.add_scalar('single_eval_loss', single_eval_loss / eval_steps, step_num)
        # save model
        os.makedirs(f'checkpoints/{FLAGS.model}_{FLAGS.dataset_name}', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/{FLAGS.model}_{FLAGS.dataset_name}/step{step_num}.pt')
