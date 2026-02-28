from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from datetime import datetime
import pytz

def embed_positions(x, time_scale):
    sin_pos = np.sin(2 * np.pi * x / time_scale)
    cos_pos = np.cos(2 * np.pi * x / time_scale)
    return sin_pos, cos_pos

fee_pct = 0.0005

class LOBEnv(gym.Env):
    def __init__(self, orderbook, device, book_depth, normalization=True):
        # large_cpu_tensor should be pinned for speed
        if device == 'cuda':
            self.data = orderbook.pin_memory() 
        else:
            self.data = orderbook
        self.device = device
        self.book_depth = book_depth
        self.normalization = normalization
        self.current_step = 0
        self.max_steps = self.data.shape[0] - 1
        self.initial_cash = 100_000.0
        
        self.action_space = spaces.Discrete(3)
        if self.normalization:
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(48,), dtype=float)
        else:
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(36,), dtype=float)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.market_open_step = 0
        self.market_open_time = None
        self.time_since_open = 0

        self.start_year = 2022 # change this to whatever year testing starts at
        self.current_year = None
        self.weekday = None
        self.current_month = None
        self.current_date = None
        self.current_ts = 0
        
        self.cash = self.initial_cash
        self.total_balance = self.initial_cash
        self.inventory = 0
        self.entry_price = 0
        self.steps_held = 0
        self.transaction_costs = 0
        self.total_return = 0

        self.volume_mismatches = 0
        
        return self._get_obs(), {}
    
    def _get_LOB_data(self, LOB):
        bid_prices = LOB[1:(self.book_depth*2)+1:2]
        bid_volumes = LOB[2:(self.book_depth*2)+1:2]

        ask_prices = LOB[(self.book_depth*2)+1:(self.book_depth*4)+1:2]
        ask_volumes = LOB[(self.book_depth*2)+2:(self.book_depth*4)+1:2]

        return bid_prices, bid_volumes, ask_prices, ask_volumes
    
    def _update_time_info(self, current_ts, current_date):
        year, hour, minute, second = current_date.year, current_date.hour, current_date.minute, current_date.second
        # if new day
        if hour == 9 and minute == 30 and second == 0:
            self.current_date = current_date
            self.market_open_time = current_ts
            self.market_open_step = self.current_step

            self.current_year = year
            self.current_month = self.current_date.month
            self.weekday = current_date.weekday()
        
        self.current_ts = current_ts.item()
        self.time_since_open = current_ts.item() - self.market_open_time

    def _create_pos_encodings(self):
        # 60 * 60 * 24 = total seconds in a day
        daily_pos_sin, daily_pos_cos = embed_positions(self.time_since_open.item(), 60*60*24)
        weekday_pos_sin, weekday_pos_cos = embed_positions(self.current_date.weekday(), 7)
        monthly_pos_sin, monthly_pos_cos = embed_positions(self.current_month, 12)
        year_pos = self.current_year - self.start_year

        return daily_pos_sin, daily_pos_cos, weekday_pos_sin, weekday_pos_cos, monthly_pos_sin, monthly_pos_cos, year_pos

    def _get_historical_info(self):
        index_10_sec_ago = max(self.current_step - 10, self.market_open_step)
        index_1_min_ago = max(self.current_step - 60, self.market_open_step)
        index_5_min_ago = max(self.current_step - 300, self.market_open_step)
        
        LOB_10s = self.data[index_10_sec_ago]
        LOB_1m = self.data[index_1_min_ago]
        LOB_5m = self.data[index_5_min_ago]

        midpoint_10s = (LOB_10s[7] + LOB_10s[1]) / 2
        midpoint_1m = (LOB_1m[7] + LOB_1m[1]) / 2
        midpoint_5m = (LOB_5m[7] + LOB_5m[1]) / 2

        midpoints = torch.stack([midpoint_10s, midpoint_1m, midpoint_5m])
        return midpoints

    def _get_basic_obs(self, LOB):
        bid_prices, bid_volumes, ask_prices, ask_volumes = self._get_LOB_data(LOB)

        current_ts = LOB[0]
        current_date = datetime.fromtimestamp(current_ts.item(), pytz.timezone("US/Eastern"))
        self._update_time_info(current_ts, current_date)

        return bid_prices, bid_volumes, ask_prices, ask_volumes

    def _get_obs(self):
        LOB = self.data[self.current_step]
        
        # get basic observational data from LOB
        bid_prices, bid_volumes, ask_prices, ask_volumes = self._get_basic_obs(LOB)

        cumu_bid_volume = sum(bid_volumes)
        cumu_ask_volume = sum(ask_volumes)
        bid_liquidity = sum(bid_prices * bid_volumes)
        ask_liquidity = sum(ask_prices * ask_volumes)
        # create features from the orderbook
        order_book_imbalance = (cumu_bid_volume - cumu_ask_volume) / (cumu_bid_volume + cumu_ask_volume)
        spread = ask_prices[0] - bid_prices[0]
        midpoint = (ask_prices[0] - bid_prices[0]) / 2
        microprice = (bid_liquidity + ask_liquidity) / (cumu_bid_volume + cumu_ask_volume)

        # get trade info
        trade_info = LOB[-3:]

        # create positonal embeddings
        daily_pos_sin, daily_pos_cos, weekday_pos_sin, weekday_pos_cos, monthly_pos_sin, monthly_pos_cos, year_pos = self._create_pos_encodings()

        # get historical data
        historical_midpoints = self._get_historical_info()
        midpoint_changes = midpoint / historical_midpoints

        # normalize existing values
        if self.normalization:
            book_prices_log_norm = torch.cat([ask_prices / midpoint, bid_prices / midpoint])
            book_volumes_norm = torch.cat([ask_volumes / cumu_ask_volume, bid_volumes / cumu_bid_volume])
            book_values_log = torch.cat([
                ask_prices,
                bid_prices,
                ask_volumes,
                bid_volumes,
                cumu_ask_volume.view(1),
                cumu_bid_volume.view(1),
                ask_liquidity.view(1),
                bid_liquidity.view(1),
                midpoint.view(1),
                microprice.view(1)
            ])

            book_prices_log_norm = torch.log1p(book_prices_log_norm)
            book_values_log = torch.log(book_values_log)
            trade_info[1] = torch.log(trade_info[1])

            LOB_norm = torch.cat([book_prices_log_norm, book_volumes_norm, book_values_log, trade_info])
        else:
            LOB = torch.cat([
                ask_prices,
                ask_volumes,
                bid_prices,
                bid_volumes,
                cumu_ask_volume.view(1),
                cumu_bid_volume.view(1),
                ask_liquidity.view(1),
                bid_liquidity.view(1),
                midpoint.view(1),
                microprice.view(1),
                trade_info
            ])

        # create tensors that will be used for the observation space
        # create tensor private state from inventory and balance
        if self.normalization:
            private_state = torch.tensor([
                self.cash / self.initial_cash,
                np.log1p(self.inventory),
                self.total_balance / self.initial_cash
                
            ], dtype=torch.float32)
        else:
            private_state = torch.tensor([
                self.cash,
                self.inventory,
                self.total_balance
                
            ], dtype=torch.float32)

        # create tensor for positional embeddings
        positional_embeddings = torch.tensor([
            daily_pos_sin, daily_pos_cos, weekday_pos_sin, weekday_pos_cos, monthly_pos_sin, monthly_pos_cos, year_pos
        ])

        # combine all tensors together
        #full_obs = torch.cat([LOB, engineered_features, private_state, positional_embeddings])
        if self.normalization:
            full_obs = torch.cat([LOB_norm, order_book_imbalance.view(1), spread.view(1), midpoint_changes, private_state, positional_embeddings])
        else:
            full_obs = torch.cat([LOB, order_book_imbalance.view(1), spread.view(1), midpoint_changes, private_state, positional_embeddings])
        
        if self.device == 'cuda':
            full_obs = full_obs.to(device=self.device, dtype=torch.float32, non_blocking=True)
        else:
            full_obs = full_obs.to(device=self.device, dtype=torch.float32)
        return full_obs
    
    def _get_midpoint(self):
        top_ask = self.data[self.current_step][(self.book_depth*2)+1]
        top_bid = self.data[self.current_step][1]
        return (top_ask + top_bid) / 2

    def step(self, action):
        # 1. get LOB data at current state and make necessary variables for the action taken
        LOB = self.data[self.current_step]

        bid_prices, bid_volumes, ask_prices, ask_volumes = self._get_basic_obs(LOB)
        midpoint = (ask_prices[0] + bid_prices[0]) / 2

        exec_shares = 0
        exec_cost = 0
        exec_fees = 0

        if action == 1: # buy order
            trade_budget = self.cash * 0.01 # 1% of portfolio value
            remaining_budget = trade_budget
            
            for i in range(len(ask_prices)):
                price = ask_prices[i].item()
                volume = ask_volumes[i].item()

                # get the maximum number of whole shares that can be traded based on budget
                max_shares = remaining_budget // (price * (1 + fee_pct))

                # take the minimum volume between what is available and wehat we can trade
                shares_taken = min(volume, max_shares)

                # compute metrics
                cost = shares_taken * price
                fees = cost * fee_pct
                exec_shares += shares_taken
                exec_cost += (cost + fees)
                exec_fees += fees
                remaining_budget -= (cost + fees)

                if remaining_budget <= 0 or shares_taken < volume:
                    break

            if exec_shares > 0:
                total_val = (self.inventory * self.entry_price) + exec_cost
                self.inventory += exec_shares
                self.entry_price = total_val / self.inventory
                self.cash -= exec_cost
                self.transaction_costs += exec_fees
            
        elif action == 2: # sell order
            trade_vol = self.inventory
            remaining_inventory = trade_vol

            for i in range(len(bid_prices)):
                price = bid_prices[i].item()
                volume = bid_volumes[i].item()

                # get the max number of shares that can be sold
                shares_sold = min(volume, remaining_inventory)

                # compute metrics
                cost = shares_sold * price
                fees = cost * fee_pct
                exec_shares += shares_sold
                exec_cost += (cost - fees)
                exec_fees += fees
                remaining_inventory -= shares_sold

                if remaining_inventory <= 0 or shares_sold < volume:
                    break
            # handle edge case where not all shares are sold. give warning and execute remaining sale at the worst bid price
            if remaining_inventory > 0:
                cost = remaining_inventory * bid_prices[-1].item()
                fees = cost * fee_pct
                exec_shares += remaining_inventory
                exec_cost += (cost - fees)
                exec_fees += fees
                remaining_inventory -= remaining_inventory
                
            if exec_shares > 0:
                self.total_return += exec_cost - (self.entry_price * self.inventory)
                self.inventory = 0
                self.entry_price = 0
                self.cash += exec_cost
                self.transaction_costs += exec_fees
                
        # 2. Increment step
        self.current_step += 1

        if self.current_step >= self.max_steps:
            terminated = True
            truncated = False
        elif self.total_balance <= 0:
            terminated = False
            truncated = True 
        else:
            terminated = False
            truncated = False

        # replace these variables with the variables from the observation space
        midpoint = self._get_midpoint()
        new_total_balance = self.cash + (self.inventory * midpoint)
        print(f'new_total_balance: cash: {self.cash} + (inventory: {self.inventory} * midpoint: {midpoint}) = {new_total_balance}')
        reward = new_total_balance - self.total_balance
        self.total_balance = new_total_balance
        print(f'reward: {reward}')
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}
    
