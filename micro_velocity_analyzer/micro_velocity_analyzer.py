import argparse
import os
import pickle
import numpy as np
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_chunk_balances(args):
    addresses, accounts_chunk, min_block_number, max_block_number, save_every_n, LIMIT, pos = args
    results = {}
    save_block_numbers = [min_block_number + i * save_every_n for i in range(LIMIT)]
    for address in tqdm(addresses, position=pos, leave=False):
        # Collect all balance changes
        balance_changes = []
        for block_number, amount in accounts_chunk[address][0].items():
            balance_changes.append((int(block_number), float(amount)))
        for block_number, amount in accounts_chunk[address][1].items():
            balance_changes.append((int(block_number), -float(amount)))
        # Sort the balance changes by block number
        balance_changes.sort()
        # Initialize balance and index for balance changes
        balance = 0.0
        change_idx = 0
        balances = []
        # Iterate over save_block_numbers
        for block_number in save_block_numbers:
            # Apply all balance changes up to the current block_number
            while change_idx < len(balance_changes) and balance_changes[change_idx][0] <= block_number:
                balance += balance_changes[change_idx][1]
                change_idx += 1
            balances.append(balance)
        results[address] = np.array(balances, dtype=np.float64)
    
    del balance_changes, balances
    return results

def process_chunk_balances_v2(args):
    addresses, accounts_chunk, min_block_number, max_block_number, save_every_n, LIMIT, pos = args
    
    save_block_numbers = [min_block_number + i * save_every_n for i in range(LIMIT)]

    results = {}

    for address in tqdm(addresses, position=pos, leave=False):
        current_balance = 0.0
        block_numbers = set(accounts_chunk[address][0].keys()).union(set(accounts_chunk[address][1].keys()))
        block_numbers = sorted(block_numbers)

        balances = np.zeros(len(save_block_numbers), dtype=np.float64)
        for block in block_numbers:
            if block in accounts_chunk[address][0]:
                current_balance += accounts_chunk[address][0][block]
            if block in accounts_chunk[address][1]:
                current_balance -= accounts_chunk[address][1][block]
            idx = (block - min_block_number) // save_every_n + 1
            if idx < len(balances):
                balances[idx] = current_balance

        results[address] = balances
        del balances
    
    return results

def process_chunk_velocities(args):
    addresses, accounts_chunk, min_block_number, save_every_n, LIMIT, pos = args
    results = {}
    for address in tqdm(addresses, position=pos, leave=False):
        if len(accounts_chunk[address][0]) > 0 and len(accounts_chunk[address][1]) > 0:
            arranged_keys = [list(accounts_chunk[address][0].keys()), list(accounts_chunk[address][1].keys())]
            arranged_keys[0].sort()
            arranged_keys[1].sort()
            ind_velocity = np.zeros(LIMIT, dtype=np.float64)

            for border in arranged_keys[1]:
                arranged_keys[0] = list(accounts_chunk[address][0].keys())
                test = np.array(arranged_keys[0], dtype=int)

                for i in range(0, len(test[test < border])):
                    counter = test[test < border][(len(test[test < border]) - 1) - i]
                    asset_amount = float(accounts_chunk[address][0][counter])
                    liability_amount = float(accounts_chunk[address][1][border])
                    if (asset_amount - liability_amount) >= 0:
                        idx_range = np.unique(np.arange(counter - min_block_number, border - min_block_number)//save_every_n)
                        if len(idx_range) == 1:
                            accounts_chunk[address][0][counter] -= liability_amount
                            accounts_chunk[address][1].pop(border)
                            break
                        else:
                            duration = border - counter
                            if duration > 0:
                                ind_velocity[idx_range[1:]] += liability_amount / duration
                            accounts_chunk[address][0][counter] -= liability_amount
                            accounts_chunk[address][1].pop(border)
                            break
                    else:
                        idx_range = np.unique(np.arange(counter - min_block_number, border - min_block_number)//save_every_n)
                        if len(idx_range) == 1:
                            accounts_chunk[address][1][border] -= asset_amount
                            accounts_chunk[address][0].pop(counter)
                        else:
                            duration = border - counter
                            if duration > 0:
                                ind_velocity[idx_range[1:]] += asset_amount / duration
                            accounts_chunk[address][1][border] -= asset_amount
                            accounts_chunk[address][0].pop(counter)
            results[address] = ind_velocity
    return results

class MicroVelocityAnalyzer:
    def __init__(self, allocated_file, transfers_file, output_file='temp/general_velocities.pickle', save_every_n=1, n_cores=1, n_chunks=1, 
                 split_save=False, batch_size=1):
        self.allocated_file = allocated_file
        self.transfers_file = transfers_file
        self.output_file = output_file
        self.save_every_n = save_every_n
        self.n_cores = n_cores
        self.n_chunks = n_chunks
        self.split_save = split_save
        self.batch_size = batch_size
        self.accounts = {}
        self.backup_accounts = {}
        self.min_block_number = float('inf')
        self.max_block_number = float('-inf')
        self.velocities = {}
        self.balances = {}
        self.LIMIT = 0
        self._create_output_folder()

    def _create_output_folder(self):
        output_folder = os.path.dirname(self.output_file)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def load_allocated_data(self):
        with open(self.allocated_file, 'r') as file:
            reader = csv.DictReader(file)
            for line in tqdm(reader):
                self._process_allocation(line)

    def _process_allocation(self, line):
        to_address = line['to_address'].lower()
        try:
            amount = float(line['amount'])  # Use float
            block_number = int(line['block_number'])
        except ValueError:
            print(f"Invalid data in allocated_file: {line}")
            return  # Skip this line

        if to_address not in self.accounts:
            self.accounts[to_address] = [{}, {}]
        
        if block_number not in self.accounts[to_address][0]:
            self.accounts[to_address][0][block_number] = amount
        else:
            self.accounts[to_address][0][block_number] += amount

        self.min_block_number = min(self.min_block_number, block_number)
        self.max_block_number = max(self.max_block_number, block_number)

    def load_transfer_data(self):
        with open(self.transfers_file, 'r') as file:
            reader = csv.DictReader(file)
            for line in tqdm(reader):
                self._process_transfer(line)

    def _process_transfer(self, line):
        from_address = line['from_address'].lower()
        to_address = line['to_address'].lower()
        try:
            amount = float(line['amount'])  # Use float
            block_number = int(line['block_number'])
        except ValueError:
            print(f"Invalid data in transfers_file: {line}")
            return  # Skip this line

        # Assets
        if to_address not in self.accounts:
            self.accounts[to_address] = [{}, {}]
        if block_number not in self.accounts[to_address][0]:
            self.accounts[to_address][0][block_number] = amount
        else:
            self.accounts[to_address][0][block_number] += amount

        # Liabilities
        if from_address not in self.accounts:
            self.accounts[from_address] = [{}, {}]
        if block_number not in self.accounts[from_address][1]:
            self.accounts[from_address][1][block_number] = amount
        else:
            self.accounts[from_address][1][block_number] += amount

        self.min_block_number = min(self.min_block_number, block_number)
        self.max_block_number = max(self.max_block_number, block_number)

    def calculate_balances(self):
        save_block_numbers = [self.min_block_number + i * self.save_every_n for i in range(self.LIMIT)]
        for address in tqdm(self.accounts.keys()):
            # Collect all balance changes
            balance_changes = []
            for block_number, amount in self.accounts[address][0].items():
                balance_changes.append((int(block_number), float(amount)))
            for block_number, amount in self.accounts[address][1].items():
                balance_changes.append((int(block_number), -float(amount)))
            # Sort the balance changes by block number
            balance_changes.sort()
            # Initialize balance and index for balance changes
            balance = 0.0
            change_idx = 0
            balances = []
            # Iterate over save_block_numbers
            for block_number in save_block_numbers:
                # Apply all balance changes up to the current block_number
                while change_idx < len(balance_changes) and balance_changes[change_idx][0] <= block_number:
                    balance += balance_changes[change_idx][1]
                    change_idx += 1
                balances.append(balance)
            self.balances[address] = np.array(balances, dtype=np.float64)
        
    # def calculate_balances_parallel(self):
    #     addresses = list(self.accounts.keys())
    #     np.random.shuffle(addresses) # Shuffle to avoid having a few addresses with many transactions in the same chunk
    #     chunk_size = max(1, len(addresses) // self.n_chunks)
    #     chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]

    #     # Process in batches of n_cores
    #     total_chunks = len(chunks)
    #     processed_chunks = 0
        
    #     with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
    #         with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
    #             while processed_chunks < total_chunks:
    #                 # Submit batch of n_cores chunks
    #                 current_batch = chunks[processed_chunks:processed_chunks + self.n_cores]
    #                 futures = []
                    
    #                 for i, chunk in enumerate(current_batch):
    #                     accounts_chunk = {address: self.accounts[address] for address in chunk}
    #                     args = (chunk, accounts_chunk, self.min_block_number, 
    #                         self.max_block_number, self.save_every_n, 
    #                         self.LIMIT, i + 1)
    #                     futures.append(executor.submit(process_chunk_balances_v2, args))
                    
    #                 # Process results as they complete
    #                 for future in as_completed(futures):
    #                     chunk_results = future.result()
    #                     self.balances.update(chunk_results)
    #                     del chunk_results
    #                     processed_chunks += 1
    #                     pbar.update(1)
                    
    #                 # Clean up
    #                 del futures

    def _get_split_filename(self, base_type, last_address):
        """Generate filename for split saves"""
        dirname = os.path.dirname(self.output_file)
        basename = os.path.splitext(os.path.basename(self.output_file))[0]
        return os.path.join(dirname, f"{basename}_{base_type}_{last_address}.pickle")

    def _save_split_results(self, results, result_type, last_address):
        """Save intermediate results to split file"""
        filename = self._get_split_filename(result_type, last_address)
        print(f'Saving {result_type} to {filename}')
        with open(filename, 'wb') as f:
            pickle.dump(results, f)

    def calculate_balances_parallel(self):
        addresses = list(self.accounts.keys())
        if not self.split_save:
            np.random.shuffle(addresses)
            
        chunk_size = max(1, len(addresses) // self.n_chunks)
        chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]
        
        total_chunks = len(chunks)
        processed_chunks = 0
        batch_results = {}
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                while processed_chunks < total_chunks:
                    current_batch = chunks[processed_chunks:processed_chunks + self.n_cores*self.batch_size]
                    futures = []
                    
                    for i, chunk in enumerate(current_batch):
                        accounts_chunk = {address: self.accounts[address] for address in chunk}
                        args = (chunk, accounts_chunk, self.min_block_number, 
                               self.max_block_number, self.save_every_n, 
                               self.LIMIT, i + 1)
                        futures.append(executor.submit(process_chunk_balances_v2, args))
                    
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        batch_results.update(chunk_results)
                        processed_chunks += 1
                        pbar.update(1)
                    
                    if self.split_save:
                        last_address = current_batch[-1][-1]
                        self._save_split_results(batch_results, 'balances', last_address)
                        batch_results = {}
                    else:
                        self.balances.update(batch_results)
                    
                    del futures

    def calculate_velocities_parallel(self):
        addresses = list(self.accounts.keys())
        if not self.split_save:
            np.random.shuffle(addresses)
            
        chunk_size = max(1, len(addresses) // self.n_chunks)
        chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]
        
        batch_results = {}
        processed_chunks = 0
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                while processed_chunks < len(chunks):
                    current_batch = chunks[processed_chunks:processed_chunks + self.n_cores*self.batch_size]
                    futures = []
                    
                    for i, chunk in enumerate(current_batch):
                        accounts_chunk = {address: self.accounts[address] for address in chunk}
                        args = (chunk, accounts_chunk, self.min_block_number, 
                               self.save_every_n, self.LIMIT, i + 1)
                        futures.append(executor.submit(process_chunk_velocities, args))
                    
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        batch_results.update(chunk_results)
                        processed_chunks += 1
                        pbar.update(1)
                    
                    if self.split_save:
                        last_address = current_batch[-1][-1]
                        self._save_split_results(batch_results, 'velocities', last_address)
                        batch_results = {}
                    else:
                        self.velocities.update(batch_results)
                    
                    del futures

    def calculate_velocities(self):
        for address in tqdm(self.accounts.keys()):
            if len(self.accounts[address][0]) > 0 and len(self.accounts[address][1]) > 0:
                self._calculate_individual_velocity(address)

    def _calculate_individual_velocity(self, address):
        arranged_keys = [list(self.accounts[address][0].keys()), list(self.accounts[address][1].keys())]
        arranged_keys[0].sort()
        arranged_keys[1].sort()
        ind_velocity = np.zeros(self.LIMIT, dtype=np.float64)

        for border in tqdm(arranged_keys[1], leave=False):
            arranged_keys[0] = list(self.accounts[address][0].keys())
            test = np.array(arranged_keys[0], dtype=int)

            for i in range(0, len(test[test < border])):
                counter = test[test < border][(len(test[test < border]) - 1) - i]
                asset_amount = float(self.accounts[address][0][counter])
                liability_amount = float(self.accounts[address][1][border])
                if (asset_amount - liability_amount) >= 0:
                    idx_range = np.unique(np.arange(counter - self.min_block_number, border - self.min_block_number)//self.save_every_n)
                    if len(idx_range) == 1:
                        self.accounts[address][0][counter] -= liability_amount
                        self.accounts[address][1].pop(border)
                        break
                    else:
                        duration = border - counter
                        if duration > 0:
                            ind_velocity[idx_range[1:]] += liability_amount / duration
                        self.accounts[address][0][counter] -= liability_amount
                        self.accounts[address][1].pop(border)
                        break
                else:
                    idx_range = np.unique(np.arange(counter - self.min_block_number, border - self.min_block_number)//self.save_every_n)
                    if len(idx_range) == 1:
                        self.accounts[address][1][border] -= asset_amount
                        self.accounts[address][0].pop(counter)
                    else:
                        duration = border - counter
                        if duration > 0:
                            ind_velocity[idx_range[1:]] += asset_amount / duration
                        self.accounts[address][1][border] -= asset_amount
                        self.accounts[address][0].pop(counter)
        self.velocities[address] = ind_velocity

    # def calculate_velocities_parallel(self):
    #     addresses = list(self.accounts.keys())
    #     np.random.shuffle(addresses) # Shuffle to distribute addresses with different number of transactions
    #     chunk_size = max(1, len(addresses) // self.n_chunks)
    #     chunks = [addresses[i:(i + chunk_size)] for i in range(0, len(addresses), chunk_size)]

    #     args_list = []
    #     for i, chunk in enumerate(chunks):
    #         accounts_chunk = {address: self.accounts[address] for address in chunk}
    #         args_list.append((chunk, accounts_chunk, self.min_block_number, self.save_every_n, self.LIMIT, i%self.n_cores+1))

    #     with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
    #         futures = [executor.submit(process_chunk_velocities, args) for args in args_list]

    #         for future in tqdm(futures, position=0):
    #             chunk_results = future.result()
    #             self.velocities.update(chunk_results)

    def save_results(self):
        if self.split_save:
            return
        else:
            with open(self.output_file, 'wb') as file:
                pickle.dump([self.backup_accounts,self.velocities, self.balances], file)

    def run_analysis(self):
        print("Loading allocated data...",  self.allocated_file)
        self.load_allocated_data()
        print("Loading transfer data...", self.transfers_file)
        self.load_transfer_data()
        print("Computing interval of ", self.save_every_n, " blocks")
        print(f"Min block number: {self.min_block_number}")
        print(f"Max block number: {self.max_block_number}")
        self.LIMIT = (self.max_block_number - self.min_block_number)//self.save_every_n + 1
        self.backup_accounts = self.accounts.copy()
        print(f"Number of blocks considered: {self.LIMIT}")
        print("Calculating balances...")
        if self.n_cores == 1:
            self.calculate_balances()
            print("Calculating velocities...")
            self.calculate_velocities()
        else:
            self.calculate_balances_parallel()
            print("Calculating velocities...")
            self.calculate_velocities_parallel()
        print("Saving results...")
        self.save_results()
        print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Micro Velocity Analyzer')
    parser.add_argument('--allocated_file', type=str, default='sampledata/sample_allocated.csv', help='Path to the allocated CSV file')
    parser.add_argument('--transfers_file', type=str, default='sampledata/sample_transfers.csv', help='Path to the transfers CSV file')
    parser.add_argument('--output_file', type=str, default='sampledata/general_velocities.pickle', help='Path to the output file')
    parser.add_argument('--save_every_n', type=int, default=1, help='Save every Nth position of the velocity array')
    parser.add_argument('--n_cores', type=int, default=1, help='Number of cores to use')
    parser.add_argument('--n_chunks', type=int, default=1, help='Number of chunks to split the data into (must be >= n_cores)')
    parser.add_argument('--split_save', action='store_true', default=False, help='Split the save into different files')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of chunks to process in a single batch')
    args = parser.parse_args()

    analyzer = MicroVelocityAnalyzer(
        allocated_file=args.allocated_file,
        transfers_file=args.transfers_file,
        output_file=args.output_file,
        save_every_n=args.save_every_n,
        n_cores=args.n_cores,
        n_chunks=args.n_chunks,
        split_save=args.split_save,
        batch_size=args.batch_size
    )
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
