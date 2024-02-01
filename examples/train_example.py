from absl import flags
import torch
from torch import nn, optim
import tqdm
from torch.utils.data import DataLoader
from rtx.data.dataset import get_oxe_dataset, TorchRLDSDataset
from robo_transformers.inference_server import InferenceServer
from tensorboardX import SummaryWriter
import os
import tensorflow as tf
import pytorch_warmup as warmup
import numpy as np
# tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_string("dataset_name", "fractal20220817_data", "Dataset name.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory.")
flags.DEFINE_list("baselines", [], "Baselines to evaluate against.")

def eval(model: torch.nn.Module, action_tokenizer, writer: SummaryWriter, step_num, eval_data_loader, criterion, device, baseline_keys=[]):
        # evaluate
    print('evaluating')
    model.eval()
    with torch.no_grad():
        eval_loss = 0
        single_eval_loss = 0
        baselines = {}
        for baseline in baseline_keys:
            baselines[baseline] = {'loss': 0, 'model': InferenceServer(baseline.split('/')[0], baseline.split('/')[1])}


        eval_steps = 0.
        for _, sample in tqdm.tqdm(enumerate(eval_data_loader)):
            if (eval_steps == 10):
                break
            eval_steps += 1
            video = (torch.permute(sample['observation']['image_primary'],(0,1,4,2,3)) / 255.0).to(device)
            instructions = sample['language_instruction']
            ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action'], device)
            out = model.run(video, instructions)

            eval_loss += criterion(out.reshape(-1, 256), ground_truth.reshape(-1,1).squeeze()).to(device)

            action_tokens = torch.max(out[:,:,:,:],-1)[1]
            single_one_hot = nn.functional.one_hot(action_tokens[:,-1,:], 256).to(device).float()
            single_eval_loss += criterion(single_one_hot.reshape(-1, 256), ground_truth[:,-1,:].reshape(-1,1).squeeze()).detach().to('cpu')

            batch_size = video.shape[0]
            n_frames = video.shape[1]

            # Log imagea and action frames in batch first sample:
            for i in range(n_frames):
                writer.add_image('image',video[0,i,:,:,:], step_num +  n_frames*eval_steps + i, dataformats='CHW')
                writer.add_text('instruction', instructions[0], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('x_gt', ground_truth[0,i,8], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('y_gt', ground_truth[0,i,9], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('z_gt', ground_truth[0,i,10], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('roll_gt', ground_truth[0,i,4], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('pitch_gt', ground_truth[0,i,5], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('yaw_gt', ground_truth[0,i,6], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('grasp_gt', ground_truth[0,i,3], step_num +  n_frames*eval_steps + i)

                writer.add_scalar('x_pred', action_tokens[0,i,8], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('y_pred', action_tokens[0,i,9], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('z_pred', action_tokens[0,i,10], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('roll_pred', action_tokens[0,i,4], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('pitch_pred', action_tokens[0,i,5], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('yaw_pred', action_tokens[0,i,6], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('grasp_pred', action_tokens[0,i,3], step_num +  n_frames*eval_steps + i)


                

            video = torch.permute(video, (0,1,3,4,2))
            for baseline in FLAGS.baselines:
                baseline_model = baselines[baseline]['model']
                batch_actions = torch.zeros((video.shape[0], 11, 256), dtype=torch.float32, device=device)
                for i in range(batch_size):
                    for j in range(n_frames):
                        out = baseline_model(image=(video[i,j,:,:,:] * 255.0).cpu().numpy(), instruction=instructions[i], save=False)
                        
                        # print(f' \n\n   {baseline} out',out)
                        out = action_tokenizer.tokenize_dict(out, device)
                        batch_actions[i,:,:] = nn.functional.one_hot(out, 256).to(device)


                        # Log action frames in batch first sample:
                        if i == 0:
                            writer.add_scalar('x_' + baseline.replace('/','_').replace('-','_'),out[8], step_num +  n_frames*eval_steps + j)
                            writer.add_scalar('y_' + baseline.replace('/','_').replace('-','_'),out[9], step_num +  n_frames*eval_steps + j)
                            writer.add_scalar('z_' + baseline.replace('/','_').replace('-','_'),out[10], step_num +  n_frames*eval_steps + j)
                            writer.add_scalar('roll_' + baseline.replace('/','_').replace('-','_'),out[4], step_num +  n_frames*eval_steps + j)
                            writer.add_scalar('pitch_' + baseline.replace('/','_').replace('-','_'),out[5], step_num +  n_frames*eval_steps + j)
                            writer.add_scalar('yaw_' + baseline.replace('/','_').replace('-','_'),out[6], step_num +  n_frames*eval_steps + j)
                            writer.add_scalar('grasp_' + baseline.replace('/','_').replace('-','_'),out[3], step_num +  n_frames*eval_steps + j)
                        # print(f' \n\n   {baseline} tokenized',out)
           
                # print(f' \n\n   {baseline} action', torch.max(batch_actions[-1,:,:],-1)[1])
                baselines[baseline]['loss'] += criterion(batch_actions.reshape(-1, 256), ground_truth[:,-1,:].reshape(-1,1).squeeze()).to('cpu').detach()
            

    writer.add_scalar('eval_loss', eval_loss / eval_steps, step_num)
    writer.add_scalar('single_eval_loss', single_eval_loss / eval_steps, step_num)

    for baseline in FLAGS.baselines:
        writer.add_scalar(f"{baseline.replace('/','_').replace('-','_')}_single_eval_loss", baselines[baseline]['loss'] / eval_steps, step_num)


def run(model: torch.nn.Module, action_tokenizer):
    writer = SummaryWriter()
    train_ds = TorchRLDSDataset(get_oxe_dataset(FLAGS.dataset_name, train=True))
    eval_ds = TorchRLDSDataset(get_oxe_dataset(FLAGS.dataset_name, train=False))
 
    train_data_loader = DataLoader(
        train_ds,
        batch_size=FLAGS.batch_size,
        num_workers=4,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        pin_memory=True,
    )

    eval_data_loader = DataLoader(
        eval_ds,
        batch_size=FLAGS.batch_size,
        num_workers=4,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        pin_memory=True,
    )

    steps_per_epoch = 5900
    warmup_period = 500
    num_steps = steps_per_epoch * FLAGS.num_epochs - warmup_period
    t0 = num_steps
    lr_min = 1e-5
    max_step = t0 + warmup_period

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.device_count()  > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.run = model.module.run
        model.train_step = model.module.train_step
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0001)
    optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=2, eta_min=lr_min)

    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

    step_num = 0
    for epoch in range(FLAGS.num_epochs):
        print(f'epoch {epoch}')
        for i, sample in tqdm.tqdm(enumerate(train_data_loader)):
            if step_num % 100 == 0:
                eval(model, action_tokenizer, writer, step_num, eval_data_loader, criterion, device, FLAGS.baselines)

            # batch, frames, height, width, channels -> batch, frames, channel, height, width
            with torch.no_grad():
                video = (torch.permute(sample['observation']['image_primary'],(0,1,4,2,3)) / 255.0).to(device)
                instructions = sample['language_instruction']
                ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action'], device=device).reshape(-1,1).squeeze()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            out = model.train_step(video, instructions).reshape(-1, 256)
            loss = criterion(out, ground_truth)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('loss', loss.to('cpu').detach().numpy(), step_num)
            del video
            del instructions
            del ground_truth
            del out
            del loss
            torch.cuda.empty_cache()
            step_num += 1

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
            if warmup_scheduler.last_step + 1 >= max_step:
                break
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step_num)
        

        # save model
        os.makedirs(f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}', exist_ok=True)
        torch.save(model.state_dict(), f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}/step{step_num}.pt')
