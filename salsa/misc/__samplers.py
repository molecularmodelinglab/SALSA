from torch.utils.data import DataLoader, RandomSampler, Sampler
from typing import Iterator, List
import copy

from salsa.constants import FAM_SIZE
# inspo ... 
# pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
class RandomFamSampler(Sampler[List[int]]):
    """
    This sampler randomly samples fam sets and puts them in a batch.
    Args:
        sampler (Sampler or Iterable): Base sampler. 
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, anc_ids: list, 
                 batch_size: int, 
                 drop_last: bool) -> None:
        self.anc_sampler = RandomSampler(anc_ids)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        i = 0
        for anc_id in self.anc_sampler:  
            fam_ids = [anc_id+i for i in range(FAM_SIZE)]
            batch.extend(fam_ids)
            i+=1
            if i % (self.batch_size) == 0:
                yield batch
                batch = []               
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.anc_sampler) // self.batch_size  
        else:
            return (len(self.anc_sampler) + self.batch_size - 1) // self.batch_size  

        
        

# https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
# class SizedCliqueSetSampler(Sampler[List[int]]):
    
#     def __init__(self, sampler: Sampler[int], 
#                  clique_map: dict,
#                  clique_batches: List[list] ) -> None:
        
#         self.sampler = sampler
#         self.clq_to_idc = copy.deepcopy(clique_map)
#         self.clique_batches = clique_batches

#     def __iter__(self) -> Iterator[List[int]]:
#         for idx in self.sampler:
#             clique_batch = self.clique_batches[idx]
#             data_batch = []
#             for clique_idx in clique_batch:
#                 data_batch.extend(self.clq_to_idc[clique_idx])
#             yield data_batch
            
#     def __len__(self) -> int:
#         return len(self.sampler)
    


    
    
# class RandomCliqueSetSampler(Sampler[List[int]]):
#     """
#     Args:
#         sampler (Sampler or Iterable): Base sampler. 
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``
#     """

#     def __init__(self, sampler: Sampler[int], 
#                  clique_map: dict,
#                  batch_size: int, 
#                  drop_last: bool) -> None:
        
#         self.sampler = sampler
#         self.clique_map = clique_map
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#     def __iter__(self) -> Iterator[List[int]]:
#         batch = []
#         i = 0
#         for idx in self.sampler:
#             augs = self.clique_map[idx]
#             batch.extend(augs)
#             i+=1
#             if i % (self.batch_size) == 0:
#                 yield batch
#                 batch = []               
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self) -> int:
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size  
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size  

#             augs = self.clique_map[idx]
#             batch.extend(augs)
#             i+=1
#             if i % (self.batch_size) == 0:
#                 yield batch
#                 batch = []               
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self) -> int:
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size  
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size  

        