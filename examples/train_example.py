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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from einops import  rearrange
tf.config.set_visible_devices([], "GPU")

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train for.")
flags.DEFINE_integer("batch_size", 2, "Batch size.")
flags.DEFINE_integer("num_warmup_steps", 1000, "Number of warmup steps.")
flags.DEFINE_integer("shuffle_buffer_size", 1000, "Shuffle buffer size.")
flags.DEFINE_integer("eval_batch_size", 1, "Eval Batch size.")
flags.DEFINE_float("lr", 1e-3, "Learning Rate.")
flags.DEFINE_float("min_lr", 1e-6, "Min Learning Rate.")
flags.DEFINE_float("weight_decay", 0, "Weight Decay.")
flags.DEFINE_string("dataset_name", "fractal20220817_data", "Dataset name.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory.")
flags.DEFINE_list("baselines", [], "Baselines to evaluate against.")
flags.DEFINE_bool("data_augmentation", True, "Whether or not to use data augmentation.")
flags.DEFINE_float("conditioning_scale", 1.0, "Scale of film conditioning. on text input.")
flags.DEFINE_float("label_smoothing", 0.0, "Label smoothing.")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return dist.get_world_size()

def is_main_process():

    return get_rank() == 0

def init_distributed():
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    

def eval(model: torch.nn.Module, action_tokenizer, writer: SummaryWriter, step_num, eval_data_loader, criterion, device, baseline_keys=[], conditioning_scale=1.0):
        # evaluate
    print('evaluating')
    model.eval()
    with torch.no_grad():
        eval_loss = 0
        future_eval_loss = 0
        eval_acc = 0
        future_eval_acc = 0
        baselines = {}
        for baseline in baseline_keys:
            baselines[baseline] = {'loss': 0, 'acc': 0, 'model': InferenceServer(baseline.split('/')[0], baseline.split('/')[1])}


        eval_steps = 0.
        for _, sample in tqdm.tqdm(enumerate(eval_data_loader)):
            if (eval_steps == 100):
                break
            video = rearrange(sample['observation']['image_primary'] / 255.0, 'b f h w c -> b f c h w').to(device) / 255.0
            instructions = sample['language_instruction']
            ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action'], device)

            outs = model.run(video, instructions, conditioning_scale)
            out_preds = torch.max(outs,-1)[1]
            

            eval_loss += criterion(rearrange(outs, 'b f a bins -> (b f a) bins'), rearrange(ground_truth, 'b f a -> (b f a)')).detach().to('cpu')
            eval_acc += (out_preds == ground_truth).float().mean().detach().to('cpu')


            future_out_one_hot = nn.functional.one_hot(out_preds[:,-1,:], 256).to(device).float()
            future_gt = ground_truth[:,-1,:]

            future_eval_loss += criterion(rearrange(future_out_one_hot, 'b a bins -> (b a) bins'), rearrange(future_gt, 'b a -> (b a)')).detach().to('cpu')
            future_eval_acc += (out_preds[:,-1,:] == future_gt).float().mean().detach().to('cpu')

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

                writer.add_scalar('x_pred', out_preds[0,i,8], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('y_pred', out_preds[0,i,9], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('z_pred', out_preds[0,i,10], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('roll_pred', out_preds[0,i,4], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('pitch_pred', out_preds[0,i,5], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('yaw_pred', out_preds[0,i,6], step_num +  n_frames*eval_steps + i)
                writer.add_scalar('grasp_pred', out_preds[0,i,3], step_num +  n_frames*eval_steps + i)


                

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
                baselines[baseline]['loss'] += criterion(batch_actions[0,:,:], future_gt[0,:]).to('cpu').detach()
                baselines[baseline]['acc'] += (torch.max(batch_actions[0,:,:],-1)[1] == future_gt[0,:]).float().mean().to('cpu').detach()
            eval_steps += 1

    writer.add_scalar('eval_loss', eval_loss / eval_steps, step_num)
    writer.add_scalar('future_eval_loss', future_eval_loss / eval_steps, step_num)
    writer.add_scalar('eval_acc', eval_acc / eval_steps, step_num)
    writer.add_scalar('future_eval_acc', future_eval_acc / eval_steps, step_num)

    for baseline in FLAGS.baselines:
        writer.add_scalar(f"{baseline.replace('/','_').replace('-','_')}_future_loss", baselines[baseline]['loss'] / eval_steps, step_num)
        writer.add_scalar(f"{baseline.replace('/','_').replace('-','_')}_future_acc", baselines[baseline]['acc'] / eval_steps, step_num)
    writer.flush()

def run(model: torch.nn.Module, action_tokenizer):
    """
    Runs the training loop.
    """
    conditioning_scale = FLAGS.conditioning_scale
    init_distributed()
    writer = None
    if is_main_process():
        writer = SummaryWriter()
    train_ds = TorchRLDSDataset(*get_oxe_dataset(FLAGS.dataset_name, train=True,  data_augmentation=FLAGS.data_augmentation, shuffle_buffer_size=FLAGS.shuffle_buffer_size), train=True, rank=get_rank(), world_size=get_world_size())
    eval_ds = None
    if is_main_process():
        eval_ds = TorchRLDSDataset(*get_oxe_dataset(FLAGS.dataset_name, train=False,  data_augmentation=False, shuffle_buffer_size=FLAGS.shuffle_buffer_size), train=False, rank=0, world_size=1)
 
    train_data_loader = DataLoader(
        train_ds,
        batch_size=FLAGS.batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        pin_memory=True,
        # sampler= DistributedSampler(dataset=train_ds, shuffle=True) if torch.cuda.device_count() > 1 else None
    )
    eval_data_loader = None
    if is_main_process():
        eval_data_loader = DataLoader(
            eval_ds,
            batch_size=FLAGS.eval_batch_size,
            num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
            pin_memory=True,
            # shuffle=True,
            # sampler= DistributedSampler(dataset=eval_ds, shuffle=False) if torch.cuda.device_count() > 1 else None
        )

    steps_per_epoch = len(train_data_loader)
    warmup_period = FLAGS.num_warmup_steps
    num_steps = steps_per_epoch * FLAGS.num_epochs - warmup_period
    t0 = num_steps // 15
    lr_min = FLAGS.min_lr
    max_step = t0*3 + warmup_period

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if torch.cuda.is_available() and torch.cuda.device_count()  > 1:
        if is_main_process():
            print(f'Using {torch.cuda.device_count()} GPUs')
        # Convert BatchNorm to SyncBatchNorm. 
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DDP(model, device_ids=[get_rank()], output_device=get_rank(), find_unused_parameters=True)

        model.run = model.module.run
        model.train_step = model.module.train_step

    criterion = nn.CrossEntropyLoss(label_smoothing=FLAGS.label_smoothing)
    if is_dist_avail_and_initialized():
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    optimizer.zero_grad()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=2, eta_min=lr_min)

    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)

    step_num = 0
    for epoch in range(FLAGS.num_epochs):
        # if torch.cuda.device_count() > 1:
        #     train_data_loader.sampler.set_epoch(epoch)
        if is_main_process():
            print(f'epoch {epoch}')
        for i, sample in tqdm.tqdm(enumerate(train_data_loader)):
            if step_num % 500 == 0:
                if is_main_process():
                    eval(model, action_tokenizer, writer, step_num, eval_data_loader, criterion, device, FLAGS.baselines, conditioning_scale)
                if torch.cuda.device_count() > 1:
                    dist.barrier()

            video = rearrange(sample['observation']['image_primary'] / 255.0, 'b f h w c -> b f c h w').to(device) / 255.0
            instructions = sample['language_instruction']
            ground_truth = action_tokenizer.tokenize_xyzrpyg(sample['action'], device)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():

            outs = model.train_step(video, instructions)
            out_preds = torch.max(outs,-1)[1]
            

            loss = criterion(rearrange(outs, 'b f a bins -> (b f a) bins'), rearrange(ground_truth, 'b f a -> (b f a)' ))
            loss.backward()
            optimizer.step()
            acc = (out_preds == ground_truth).float().mean().detach().to('cpu')
            # mixed precision training 
            # backward + optimizer step
            # fp16_scaler.scale(loss).backward()
            # fp16_scaler.step(optimizer)
            # fp16_scaler.update()
            if is_main_process():
                writer.add_scalar('loss', float(loss.to('cpu').detach().numpy()), step_num)
                writer.add_scalar('acc', float(acc.to('cpu').detach().numpy()), step_num)
            # del video
            # del instructions
            # del ground_truth
            # del out
            # del loss
            # torch.cuda.empty_cache()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    lr_scheduler.step()
            if warmup_scheduler.last_step + 1 >= max_step:
                break
            if is_main_process():
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step_num)
            
            if (step_num + 1) % 1000 == 0:
                # save model
                if is_main_process():
                    os.makedirs(f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}', exist_ok=True)
                    if is_dist_avail_and_initialized():
                        torch.save(model.module.state_dict(), f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}/step{step_num}.pt')
                    else:
                         torch.save(model.state_dict(), f'{FLAGS.checkpoint_dir}/{FLAGS.model}_{FLAGS.dataset_name}/step{step_num}.pt')
            step_num += 1
