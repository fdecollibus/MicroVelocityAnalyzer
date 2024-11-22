import argparse
import os
import pickle
import numpy as np
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

class MicroVelocityAnalyzer:
    def __init__(self, allocated_file, transfers_file, output_file='temp/general_velocities.pickle', save_every_n=1, n_cores=1):
        self.allocated_file = allocated_file
        self.transfers_file = transfers_file
        self.output_file = output_file
        self.save_every_n = save_every_n
        self.n_cores = n_cores
        self.accounts = {}
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
        amount = int(line['amount'])
        block_number = int(line['block_number'])

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
        amount = int(line['amount'])
        block_number = int(line['block_number'])

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

    def calculate_velocities(self):
        for address in tqdm(self.accounts.keys()):
            if len(self.accounts[address][0]) > 0 and len(self.accounts[address][1]) > 0:
                self._calculate_individual_velocity(address)

    def _calculate_individual_velocity(self, address):
        arranged_keys = [list(self.accounts[address][0].keys()), list(self.accounts[address][1].keys())]
        arranged_keys[0].sort()
        arranged_keys[1].sort()
        ind_velocity = np.zeros(self.LIMIT)

        for border in tqdm(arranged_keys[1], leave=False):
            arranged_keys[0] = list(self.accounts[address][0].keys())
            test = np.array(arranged_keys[0])

            for i in range(0, len(test[test < border])):
                counter = test[test < border][(len(test[test < border]) - 1) - i]
                if (self.accounts[address][0][counter] - self.accounts[address][1][border]) >= 0:
                    idx_range = np.unique(np.arange(counter-self.min_block_number, border-self.min_block_number)//self.save_every_n)
                    if len(idx_range) == 1:
                        self.accounts[address][0][counter] -= self.accounts[address][1][border]
                        self.accounts[address][1].pop(border)
                        break
                    else:
                        #ind_velocity[(counter-self.min_block_number):(border-self.min_block_number)] += (self.accounts[address][1][border]) / (border - counter)
                        ind_velocity[idx_range[1:]] += (self.accounts[address][1][border]) / (border - counter)
                        #print(ind_velocity[(counter-self.min_block_number):(border-self.min_block_number)])
                        self.accounts[address][0][counter] -= self.accounts[address][1][border]
                        self.accounts[address][1].pop(border)
                        break
                else:
                    idx_range = np.unique(np.arange(counter-self.min_block_number, border-self.min_block_number)//self.save_every_n)
                    if len(idx_range) == 1:
                        self.accounts[address][1][border] -= self.accounts[address][0][counter]
                        self.accounts[address][0].pop(counter)
                    else:
                        #ind_velocity[counter-self.min_block_number:border-self.min_block_number] += (self.accounts[address][0][counter]) / (border - counter)
                        ind_velocity[idx_range[1:]] += (self.accounts[address][0][counter]) / (border - counter)
                        #print(ind_velocity[(counter-self.min_block_number):(border-self.min_block_number)])
                        self.accounts[address][1][border] -= self.accounts[address][0][counter]
                        self.accounts[address][0].pop(counter)
        # Save only every Nth position of the array
        self.velocities[address] = ind_velocity#[::self.save_every_n]

    def calculate_velocities_parallel(self):

        # Split addresses into chunks
        addresses = list(self.accounts.keys())
        chunk_size = max(1, len(addresses) // self.n_cores)
        chunks = [addresses[i:i + chunk_size] for i in range(0, len(addresses), chunk_size)]

        # Process chunks in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._process_chunk, chunk, i) for i,chunk in enumerate(chunks)]
            
            # Collect results
            for future in tqdm(futures):
                chunk_results = future.result()
                self.velocities.update(chunk_results)

    # Helper function to process chunks
    def _process_chunk(self, addresses, i):
        results = {}
        for address in tqdm(addresses, position=i, leave=False):
            if len(self.accounts[address][0]) > 0 and len(self.accounts[address][1]) > 0:
                arranged_keys = [list(self.accounts[address][0].keys()), list(self.accounts[address][1].keys())]
                arranged_keys[0].sort()
                arranged_keys[1].sort()
                ind_velocity = np.zeros(self.LIMIT)

                for border in arranged_keys[1]:
                    arranged_keys[0] = list(self.accounts[address][0].keys())
                    test = np.array(arranged_keys[0])

                    for i in range(0, len(test[test < border])):
                        counter = test[test < border][(len(test[test < border]) - 1) - i]
                        if (self.accounts[address][0][counter] - self.accounts[address][1][border]) >= 0:
                            idx_range = np.unique(np.arange(counter-self.min_block_number, border-self.min_block_number)//self.save_every_n)
                            if len(idx_range) == 1:
                                self.accounts[address][0][counter] -= self.accounts[address][1][border]
                                self.accounts[address][1].pop(border)
                                break
                            else:
                                #ind_velocity[(counter-self.min_block_number):(border-self.min_block_number)] += (self.accounts[address][1][border]) / (border - counter)
                                ind_velocity[idx_range[1:]] += (self.accounts[address][1][border]) / (border - counter)
                                #print(ind_velocity[(counter-self.min_block_number):(border-self.min_block_number)])
                                self.accounts[address][0][counter] -= self.accounts[address][1][border]
                                self.accounts[address][1].pop(border)
                                break
                        else:
                            idx_range = np.unique(np.arange(counter-self.min_block_number, border-self.min_block_number)//self.save_every_n)
                            if len(idx_range) == 1:
                                self.accounts[address][1][border] -= self.accounts[address][0][counter]
                                self.accounts[address][0].pop(counter)
                            else:
                                #ind_velocity[counter-self.min_block_number:border-self.min_block_number] += (self.accounts[address][0][counter]) / (border - counter)
                                ind_velocity[idx_range[1:]] += (self.accounts[address][0][counter]) / (border - counter)
                                #print(ind_velocity[(counter-self.min_block_number):(border-self.min_block_number)])
                                self.accounts[address][1][border] -= self.accounts[address][0][counter]
                                self.accounts[address][0].pop(counter)
                results[address] = ind_velocity#[::self.save_every_n]
        return results

    def calculate_balances(self):
        for address in tqdm(self.accounts.keys()):
            balance = 0
            balances = np.zeros(self.LIMIT)
            for block_number in tqdm(range(self.min_block_number, self.max_block_number + 1), leave=False):
                if block_number in self.accounts[address][0]:
                    balance += self.accounts[address][0][block_number]
                if block_number in self.accounts[address][1]:
                    balance -= self.accounts[address][1][block_number]
                if block_number % self.save_every_n == 0:
                    balances[(block_number - self.min_block_number)//self.save_every_n] = balance
            # Save only every Nth position of the array
            self.balances[address] = balances#[::self.save_every_n]

    def save_results(self):
        with open(self.output_file, 'wb') as file:
            pickle.dump([self.velocities, self.balances], file)

    def run_analysis(self):
        print("Loading allocated data...",  self.allocated_file)
        self.load_allocated_data()
        print("Loading transfer data...", self.transfers_file)
        self.load_transfer_data()
        print ("Computing interval of ", self.save_every_n, " blocks")
        print(f"Min block number: {self.min_block_number}")
        print(f"Max block number: {self.max_block_number}")
        self.LIMIT = (self.max_block_number - self.min_block_number)//self.save_every_n + 1

        print(f"Number of blocks considered: {self.LIMIT}")
        print("Calculating velocities...")
        if self.n_cores == 1:
            self.calculate_velocities()
        else:
            self.calculate_velocities_parallel()
        print("Calculating balances...")
        self.calculate_balances()
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
    args = parser.parse_args()

    analyzer = MicroVelocityAnalyzer(
        allocated_file=args.allocated_file,
        transfers_file=args.transfers_file,
        output_file=args.output_file,
        save_every_n=args.save_every_n,
        n_cores=args.n_cores
    )
    analyzer.run_analysis()

if __name__ == "__main__":
    main()