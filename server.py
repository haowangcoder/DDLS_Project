import torch
import re
from typing import List, Dict


def gather_cluster_signatures(client_signatures, lora_client_map) -> dict[int, torch.Tensor]:
    """Average available client signatures within each LoRA cluster."""
    cluster_signatures = {}

    for raw_cluster_id, client_ids in lora_client_map.items():
        signatures = []
        for client_id in client_ids:
            signature = client_signatures.get(int(client_id))
            if signature is not None:
                signatures.append(signature.detach().float().cpu())

        if signatures:
            cluster_signatures[int(raw_cluster_id)] = torch.stack(signatures, dim=0).mean(dim=0)

    return cluster_signatures


class Server:
    def __init__(self, clients_num: int, device: str = "cuda"):
        self.clients_num = clients_num
        self.device = device
        self.lora_client_map = None  

    def no_aggregation(self, params: List) -> List[Dict]:
        """Identity aggregation: each client keeps its own params."""
        return [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

    def aggregation_warmup(self, route_aggregation: bool, params: List, lora_client_map=None) -> List[Dict]:
        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]

        final_warmup_round = lora_client_map is not None

        if final_warmup_round:
            self.lora_client_map = lora_client_map
            print("Final warmup round, preparing transition to clustered LoRA")

            for client_idx in range(num_clients):
                for param_name, param_value in gpu_params[client_idx].items():
                    aggregated_results[client_idx][param_name] = param_value

            client_to_group = {}
            for group_idx, clients in lora_client_map.items():
                for client in clients:
                    client_to_group[client] = int(group_idx)

            for group_idx, group_clients in lora_client_map.items():
                group_idx = int(group_idx)

                if not group_clients:
                    continue

                print(f"Processing group {group_idx} with clients {group_clients}")

                valid_clients = [c for c in group_clients if c < num_clients]

                if not valid_clients:
                    continue

                for base_param_name in list(gpu_params[0].keys()):
                    if 'lora_A0' in base_param_name or 'lora_B0' in base_param_name:
                        target_param_name = base_param_name.replace('0', str(group_idx))

                        try:
                            stacked_params = torch.stack([
                                gpu_params[i][base_param_name]
                                for i in valid_clients if base_param_name in gpu_params[i]
                            ]).to(self.device)

                            if stacked_params.size(0) > 0:
                                avg_param = stacked_params.mean(dim=0)

                                for client_idx in group_clients:
                                    if client_idx < num_clients:
                                        aggregated_results[client_idx][target_param_name] = avg_param
                        except Exception as e:
                            print(f"Error aggregating {base_param_name} for group {group_idx}: {e}")
        else:
            for client_idx in range(num_clients):
                for param_name, param_value in gpu_params[client_idx].items():
                    if 'lora_A' in param_name or 'lora_B' in param_name or 'lora_route' in param_name:
                        aggregated_results[client_idx][param_name] = param_value

        return aggregated_results
    
    def aggregation(
        self,
        route_aggregation: bool,
        params: List,
        lora_client_map=None,
        universal_idx=None,
        soft_membership=None,
        shared_lora_a=False,
        fedrod_dual_head=False,
        visa_conflict_clip=False,
    ) -> List[Dict]:
        if lora_client_map is not None:
            self.lora_client_map = lora_client_map

        if self.lora_client_map is None:
            raise ValueError("lora_client_map must be provided for aggregation after warmup phase")

        client_to_group = {}
        for group_idx, clients in self.lora_client_map.items():
            for client in clients:
                client_to_group[client] = group_idx

        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]
        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]
        param_names = gpu_params[0].keys()

        for client_idx in range(num_clients):
            for param_name in param_names:

                if fedrod_dual_head and _is_fedrod_global_param(param_name):
                    group_indices = list(range(num_clients))
                    stacked_params = torch.stack([
                        gpu_params[i][param_name]
                        for i in group_indices if param_name in gpu_params[i]
                    ]).to(self.device)
                    aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)

                elif fedrod_dual_head and _is_classifier_param(param_name):
                    client_group = client_to_group.get(client_idx)
                    if client_group is not None:
                        group_indices = self.lora_client_map[client_group]
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in group_indices if i < len(gpu_params) and param_name in gpu_params[i]
                        ]).to(self.device)
                        if stacked_params.size(0) > 0:
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

                elif 'lora_route' in param_name:
                    if route_aggregation:
                        client_group = client_to_group.get(client_idx)
                        if client_group is not None:
                            group_indices = self.lora_client_map[client_group]
                            stacked_params = torch.stack([
                                gpu_params[i][param_name]
                                for i in group_indices
                            ]).to(self.device)
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

                elif 'lora_A' in param_name or 'lora_B' in param_name:
                    prefix = "lora_A" if "lora_A" in param_name else "lora_B"
                    lora_idx = _extract_lora_index(param_name, prefix)

                    # HydraFed-LEASE: lora_A pooled across ALL clients regardless of cluster
                    # MUST be checked BEFORE the soft-membership branch. Under TRUE PEFT
                    # parameter tying (peft/tuners/lora.py shared_lora_a=True), only
                    # lora_A0.weight is exported (named_parameters dedups the tied tensor).
                    # If we let soft membership win first, lora_A0 would be aggregated as
                    # "cluster 0" using soft_membership[*][0] weights, biasing the global
                    # tied A toward whichever cluster is index 0 (e.g., SST-2 under
                    # oracle assignment with the canonical task ordering).
                    if shared_lora_a and prefix == "lora_A" and lora_idx != universal_idx:
                        group_indices = list(range(num_clients))
                        if group_indices:
                            stacked_params = torch.stack([
                                gpu_params[i][param_name]
                                for i in group_indices if i < len(gpu_params) and param_name in gpu_params[i]
                            ]).to(self.device)
                            if stacked_params.size(0) > 0:
                                aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                            else:
                                aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                        continue

                    if soft_membership is not None and lora_idx != universal_idx:
                        weights = []
                        indices = []
                        for src_client in range(num_clients):
                            weight = soft_membership.get(src_client, {}).get(lora_idx, 0.0)
                            if weight > 0 and param_name in gpu_params[src_client]:
                                weights.append(float(weight))
                                indices.append(src_client)

                        if weights:
                            weight_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32)
                            weight_tensor = weight_tensor / weight_tensor.sum()
                            selected_params = [
                                gpu_params[i][param_name] for i in indices
                            ]
                            if visa_conflict_clip and prefix == "lora_B":
                                selected_params = _clip_conflicting_visa_params(
                                    selected_params=selected_params,
                                    indices=indices,
                                    param_name=param_name,
                                    lora_idx=lora_idx,
                                    gpu_params=gpu_params,
                                    lora_client_map=self.lora_client_map,
                                    soft_membership=soft_membership,
                                    device=self.device,
                                )
                            stacked_params = torch.stack(selected_params).to(self.device)
                            view = (-1,) + (1,) * (stacked_params.ndim - 1)
                            aggregated = (stacked_params.float() * weight_tensor.view(*view)).sum(dim=0)
                            aggregated_results[client_idx][param_name] = aggregated.to(stacked_params.dtype)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                        continue

                    group_indices = _resolve_group_indices(self.lora_client_map, lora_idx, universal_idx)

                    if group_indices:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in group_indices if i < len(gpu_params) and param_name in gpu_params[i]
                        ]).to(self.device)
                        if stacked_params.size(0) > 0:
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                else:
                    aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

        return aggregated_results


def _extract_lora_index(param_name: str, prefix: str) -> int:
    match = re.search(rf"{prefix}(\d+)", param_name)
    if match is None:
        raise ValueError(f"Could not extract LoRA index from parameter name: {param_name}")
    return int(match.group(1))


def _is_fedrod_global_param(param_name: str) -> bool:
    # alpha_logit is shape [lora_num-1] — same across all clients, safe to FedAvg globally.
    # cls_universal.* has shape [num_labels, hidden] — clients may differ on num_labels
    # (e.g., MNLI=3 vs SST-2=2 in extended-GLUE). Treat cls_universal as per-cluster
    # (oracle clusters group same-num_labels clients together).
    return param_name == "alpha_logit"


def _is_classifier_param(param_name: str) -> bool:
    # classifier (cls_E) AND cls_universal both per-cluster aggregated.
    # Per-cluster works because oracle clustering groups clients with identical num_labels.
    return "classifier" in param_name or param_name.startswith("cls_universal.")


def _clip_conflicting_visa_params(
    selected_params,
    indices,
    param_name,
    lora_idx,
    gpu_params,
    lora_client_map,
    soft_membership,
    device,
    delta=1e-12,
):
    client_to_group = _build_client_to_group(lora_client_map)
    home_clients = [
        client_idx for client_idx in indices
        if client_to_group.get(client_idx) == int(lora_idx)
    ]
    if not home_clients:
        return selected_params

    home_weights = [
        float(soft_membership.get(client_idx, {}).get(lora_idx, 0.0))
        for client_idx in home_clients
    ]
    if sum(home_weights) <= 0:
        return selected_params

    home_weight_tensor = torch.tensor(home_weights, device=device, dtype=torch.float32)
    home_weight_tensor = home_weight_tensor / home_weight_tensor.sum()
    home_stack = torch.stack([
        gpu_params[client_idx][param_name]
        for client_idx in home_clients
    ]).to(device)
    view = (-1,) + (1,) * (home_stack.ndim - 1)
    home_mean = (home_stack.float() * home_weight_tensor.view(*view)).sum(dim=0)
    home_flat = home_mean.reshape(-1)
    denom = home_flat.dot(home_flat) + float(delta)

    clipped_params = []
    for client_idx, param in zip(indices, selected_params):
        if client_to_group.get(client_idx) == int(lora_idx):
            clipped_params.append(param)
            continue

        visa_param = param.to(device)
        dot = visa_param.float().reshape(-1).dot(home_flat)
        if dot.item() < 0:
            safe_param = visa_param.float() - (dot / denom) * home_mean
            clipped_params.append(safe_param.to(dtype=visa_param.dtype))
        else:
            clipped_params.append(param)

    return clipped_params


def _build_client_to_group(lora_client_map):
    client_to_group = {}
    for group_idx, clients in lora_client_map.items():
        for client in clients:
            client_to_group[int(client)] = int(group_idx)
    return client_to_group


def _resolve_group_indices(lora_client_map, lora_idx, universal_idx=None):
    if universal_idx is not None and lora_idx == universal_idx:
        return sorted({
            client_idx
            for group_indices in lora_client_map.values()
            for client_idx in group_indices
        })

    group_indices = lora_client_map.get(str(lora_idx), [])
    if not group_indices:
        group_indices = lora_client_map.get(lora_idx, [])
    return group_indices
