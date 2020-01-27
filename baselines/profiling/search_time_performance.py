"""
Compute search time needed for searching 100 new queries in a corpus containing 1M videos.
The performance reported is tested on 1.4.0.dev20191109 with Python3.7 and CUDA10.1.

This experiment is simulated.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.basic_utils import save_json

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

np.random.seed(1234)


def compare_l2dist_inner_product_time(n_videos=2000, d=256, n_query=1000, n_runs=10, n_warmup_runs=10):
    """In some PyTorch/Cuda Verison, torch.cdist is very slow, which affects this comparison.
    See https://discuss.pytorch.org/t/cdist-vs-matmul/61682/5"""
    torch.cuda.synchronize()
    st_time = time.time()
    fake_database = F.normalize(torch.randn((n_videos, d), dtype=torch.float32).cuda(), dim=1, p=2)
    fake_query = F.normalize(torch.randn((n_query, d), dtype=torch.float32).cuda(), dim=1, p=2)
    torch.cuda.synchronize()
    print("Construct fake database + query time {}".format(time.time() - st_time))
    print("fake_database shape {} fake_query shape {}".format(fake_database.shape, fake_query.shape))

    times_l2dist = []
    for _ in range(n_warmup_runs + n_runs):
        torch.cuda.synchronize()
        st_time = time.time()
        l2_dist = torch.cdist(fake_query, fake_database, p=2)  # (n_query, n_videos)
        torch.cuda.synchronize()
        times_l2dist.append(time.time() - st_time)
    avg_time_l2dist = np.mean(times_l2dist[n_warmup_runs:])
    print("L2 Distance time {}".format(avg_time_l2dist))

    times_ip = []
    fake_database = fake_database.transpose(0, 1)
    for _ in range(n_warmup_runs + n_runs):
        torch.cuda.synchronize()
        st_time = time.time()
        inner_product = torch.mm(fake_query, fake_database)  # (n_query, n_videos)
        torch.cuda.synchronize()
        times_ip.append(time.time() - st_time)
    avg_time_ip = np.mean(times_ip[n_warmup_runs:])
    print("Inner Product time {}".format(avg_time_ip))


def run_example():
    """
    In Python, the matrices are always represented as numpy arrays.
    The data type dtype must be float32.
    """
    # --------------------------------
    # Step 1: Get Data
    # --------------------------------
    import faiss
    d = 64  # dimension
    nb = 100000  # database size
    nq = 10000  # nb of queries
    np.random.seed(1234)  # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    # --------------------------------
    # Step 2: Build `Index' object
    # Note some of the indexes require a training phase to analyze the data distribution.
    # --------------------------------
    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    index.add(xb)  # add vectors to the index
    print(index.ntotal)

    k = 4  # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k)  # sanity check
    print(I)
    print(D)
    st_time = time.time()
    D, I = index.search(xq, k)  # actual search
    print("time elapsed {}".format(time.time() - st_time))
    print(I[:5])  # neighbors of the 5 first queries
    print(I[-5:])  # neighbors of the 5 last queries


def simulate_mee_runtime(n_videos=1000000, d=256, n_query=100, max_neighbors=100, n_runs=5, n_warmup_runs=10):
    """ Search over a database of shape [n_videos, d] with query of shape [n_query, d].
    For each query, return max_neighbors results.
    """
    import faiss
    torch.cuda.synchronize()
    st_time = time.time()
    fake_database = faiss.rand((n_videos, d))
    fake_query = faiss.rand((n_query, d))
    torch.cuda.synchronize()
    logger.info("Construct fake database + query time {}".format(time.time() - st_time))

    torch.cuda.synchronize()
    st_time = time.time()
    index = faiss.index_factory(d, "IVF4096,Flat", faiss.METRIC_L2)
    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(d))
    index_ivf.clustering_index = clustering_index
    torch.cuda.synchronize()
    logger.info("Build/Move to GPU? index time {}".format(time.time() - st_time))

    st_time = time.time()
    torch.cuda.synchronize()
    index_ivf.train(fake_database)
    torch.cuda.synchronize()
    logger.info("Train index time {}".format(time.time() - st_time))

    times = []
    for _ in range(n_warmup_runs+n_runs):
        torch.cuda.synchronize()
        st_time = time.time()
        D, I = index_ivf.search(fake_query, max_neighbors)
        torch.cuda.synchronize()
        times.append(time.time() - st_time)
    avg_time = np.mean(times[n_warmup_runs:]) * 2  # video + sub
    logger.info("Avg searching time ({} runs) {}".format(n_runs, avg_time))
    return avg_time


def simulate_cal_rerank_time(n_moments=200, avg_n_clips_per_moment=7, d=256, n_query=100, max_neighbors=100,
                             n_runs=5, n_warmup_runs=10):
    st_time = time.time()
    torch.cuda.synchronize()
    fake_database = torch.randn((n_moments * avg_n_clips_per_moment, d), dtype=torch.float32).cuda()
    fake_query = torch.randn((n_query, d), dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    logger.info("Construct fake database + query time {}".format(time.time() - st_time))

    times = []
    for _ in range(n_warmup_runs+n_runs):
        torch.cuda.synchronize()
        st_time = time.time()
        fake_dist = torch.cdist(fake_query, fake_database, p=2)
        fake_dist = fake_dist.view(n_query, n_moments, avg_n_clips_per_moment).mean(2)
        fake_dist = torch.cdist(fake_query, fake_database, p=2)
        fake_dist = fake_dist.view(n_query, n_moments, avg_n_clips_per_moment).mean(2)  # video + sub
        fake_dist = fake_dist + fake_dist
        fake_top_indices, fake_top_dist = torch.topk(fake_dist, k=max_neighbors, dim=1, largest=False, sorted=True)
        torch.cuda.synchronize()
        times.append(time.time() - st_time)
    avg_time = np.mean(times[n_warmup_runs:])
    logger.info("searching time {}".format(avg_time))
    return avg_time


def simulate_mcn_rerank_time(n_moments=200, d=256, n_query=100, max_neighbors=100, n_runs=5, n_warmup_runs=10):
    torch.cuda.synchronize()
    st_time = time.time()
    fake_database = torch.randn((n_moments, d), dtype=torch.float32).cuda()
    fake_query = torch.randn((n_query, d), dtype=torch.float32).cuda()
    torch.cuda.synchronize()
    logger.info("Construct fake database + query time {}".format(time.time() - st_time))

    times = []
    for _ in range(n_warmup_runs+n_runs):
        torch.cuda.synchronize()
        st_time = time.time()
        fake_dist = torch.cdist(fake_query, fake_database, p=2).view(n_query, n_moments)
        fake_dist = torch.cdist(fake_query, fake_database, p=2).view(n_query, n_moments)  # video + sub
        fake_dist = fake_dist + fake_dist
        fake_top_indices, fake_top_dist = torch.topk(fake_dist, k=max_neighbors, dim=1, largest=False, sorted=True)
        torch.cuda.synchronize()
        times.append(time.time() - st_time)
    avg_time = np.mean(times[n_warmup_runs:])  #
    logger.info("searching time {}".format(avg_time))
    return avg_time


def simulate_xml_rerank_time(n_videos=100, avg_n_clips_per_video=20, d=256, n_query=100, max_neighbors=100,
                             n_runs=5, n_warmup_runs=10):
    torch.cuda.synchronize()
    st_time = time.time()
    fake_database = torch.randn((d, n_videos*avg_n_clips_per_video), dtype=torch.float32).cuda()
    fake_query = torch.randn((n_query, d), dtype=torch.float32).cuda()
    conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=2, bias=False).cuda()
    torch.cuda.synchronize()
    logger.info("Construct fake database + query time {}".format(time.time() - st_time))

    times = dict(
        conv=[],
        prod=[],
        topk=[],
        triu=[]
    )
    for _ in range(n_warmup_runs+n_runs):
        torch.cuda.synchronize()
        st_time = time.time()  # [100, 256] [100, 20, 256]
        fake_dist = torch.mm(fake_query, fake_database).view(n_query*n_videos, -1)
        fake_dist = torch.mm(fake_query, fake_database).view(n_query * n_videos, -1)  # video + sub
        fake_dist = fake_dist + fake_dist
        torch.cuda.synchronize()
        times["prod"].append(time.time() - st_time)
        torch.cuda.synchronize()
        st_time = time.time()
        fake_dist = conv(fake_dist.unsqueeze(1))[:, 0, :]
        torch.cuda.synchronize()
        times["conv"].append(time.time() - st_time)
        torch.cuda.synchronize()
        st_time = time.time()
        fake_prob_prod = torch.triu(torch.einsum("ns,ne->nse", fake_dist, fake_dist)).view(n_query, -1)
        torch.cuda.synchronize()
        times["triu"].append(time.time() - st_time)
        torch.cuda.synchronize()
        st_time = time.time()
        fake_top_indices, fake_top_dist = torch.topk(fake_prob_prod, k=max_neighbors, dim=1, largest=True, sorted=True)
        torch.cuda.synchronize()
        times["topk"].append(time.time() - st_time)
    avg_time = {k: np.mean(times[k][n_warmup_runs:]) for k in times}
    avg_time["all"] = np.sum(list(avg_time.values()))
    logger.info("searching time {}".format(avg_time))
    return avg_time


def get_storage_size(hsz, n_videos, n_clips_per_video, n_moments, n_total_clips_in_moments, dtype_size=4):
    """dtype_size: float32, 4B"""
    GB = 1024**3
    # multiply by 2 for video+sub, xml has two level, so it has an additional 2 to multiply by.
    storage = dict(
        mee=n_videos * hsz * dtype_size * 2. / GB,
        cal=n_total_clips_in_moments * hsz * dtype_size * 2. / GB,
        mcn=n_moments * hsz * dtype_size * 2. / GB,
        xml=n_videos * n_clips_per_video * hsz * dtype_size * 2. * 2. / GB
    )
    print("storage (GB) {}".format(storage))
    return storage


def main_run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="mee", help="which models to simulate")
    parser.add_argument("--cache_dir", type=str, default="baselines/profiling/cache", help="save index/results path")
    parser.add_argument("--n_runs", type=int, default=100, help="number of runs to calc average")
    parser.add_argument("--n_warmup_runs", type=int, default=10, help="number of warmup runs, to init cuda, etc.")
    args = parser.parse_args()

    """
    The numbers are get from the first author of 
    `Temporal Localization of Moments in Video Collections with Natural Language`
    """
    k = 100
    n_query = 100
    n_videos = 1000000
    n_moments_per_video = 170
    hsz = 256
    n_clips_per_video = 20
    n_total_clips_in_moments = 1170946944
    n_moments = 170000000
    max_clips_per_proposal = 14  # assume padding to this number
    avg_clips_per_proposal = 7  # 6.88

    mode = args.mode
    cfg_path = os.path.join(args.cache_dir, "{}_args.json".format(mode))

    n_runs = args.n_runs
    n_warmup_runs = args.n_warmup_runs
    torch.set_grad_enabled(False)
    if mode in ["mee", "mee_torch"]:
        func_args = dict(n_videos=n_videos, d=hsz, n_query=n_query, max_neighbors=k,
                         n_runs=n_runs, n_warmup_runs=n_warmup_runs)
        avg_time = simulate_mee_runtime(**func_args)
    elif mode == "xml_vr":
        func_args = dict(n_videos=n_videos*n_clips_per_video, d=hsz, n_query=n_query,
                         max_neighbors=k, n_runs=n_runs, n_warmup_runs=n_warmup_runs)
        avg_time = simulate_mee_runtime(**func_args)
    elif mode == "cal":
        # can only use n_query <= 4000, so use 4000. To get 20000, simply x5 the final time.
        n_cal_rerank_videos = 100
        func_args = dict(n_moments=n_cal_rerank_videos*n_moments_per_video,
                         avg_n_clips_per_moment=avg_clips_per_proposal,
                         d=hsz, n_query=n_query, max_neighbors=k, n_runs=n_runs, n_warmup_runs=n_warmup_runs)
        avg_time = simulate_cal_rerank_time(**func_args)
    elif mode == "mcn":
        n_cal_rerank_videos = 100
        func_args = dict(n_moments=n_cal_rerank_videos*n_moments_per_video, d=hsz, n_query=n_query,
                         max_neighbors=k, n_runs=n_runs, n_warmup_runs=n_warmup_runs)
        avg_time = simulate_mcn_rerank_time(**func_args)
    elif mode == "xml":
        n_xml_videos = 100
        func_args = dict(n_videos=n_xml_videos, avg_n_clips_per_video=n_clips_per_video,
                         d=hsz, n_query=n_query, max_neighbors=k, n_runs=n_runs, n_warmup_runs=n_warmup_runs)
        avg_time = simulate_xml_rerank_time(**func_args)
    elif mode == "storage":
        func_args = dict(hsz=hsz, n_videos=n_videos, n_clips_per_video=n_clips_per_video,
                         n_moments=n_moments, n_total_clips_in_moments=n_total_clips_in_moments, dtype_size=4)
        storage = get_storage_size(**func_args)
    else:
        raise NotImplementedError

    if mode == "storage":
        func_args["storage"] = storage
    else:
        func_args["n_runs"] = args.n_runs
        func_args["avg_time"] = avg_time
    func_args["mode"] = mode
    print(func_args)
    save_json(func_args, cfg_path, save_pretty=True)


if __name__ == '__main__':
    main_run()
    # compare_l2dist_inner_product_time()
