import csv
import warnings
import os
import pickle
import numpy as np
import argparse
from glob import glob
from setuptools import setup, find_packages

warnings.filterwarnings("ignore")

class MicroVelocityAnalyzer:
    def __init__(self, allocated_file, transfers_file, output_folder='temp', save_every_n=1):
        self.allocated_file = allocated_file
        self.transfers_file = transfers_file
        self.output_folder = output_folder
        self.save_every_n = save_every_n
        self.accounts = {}
        self.min_block_number = float('inf')
        self.max_block_number = float('-inf')
        self.velocities = {}
        self.LIMIT = 0
        self._create_output_folder()

    def _create_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_allocated_data(self):
        with open(self.allocated_file, 'r') as file:
            reader = csv.DictReader(file)
            for line in reader:
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
            for line in reader:
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
        self.LIMIT = self.max_block_number - self.min_block_number
        for address in self.accounts.keys():
            if len(self.accounts[address][0]) > 0 and len(self.accounts[address][1]) > 0:
                self._calculate_individual_velocity(address)

    def _calculate_individual_velocity(self, address):
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
                    ind_velocity[counter:border] += (self.accounts[address][1][border]) / (border - counter)
                    self.accounts[address][0][counter] -= self.accounts[address][1][border]
                    self.accounts[address][1].pop(border)
                    break
                else:
                    ind_velocity[counter:border] += (self.accounts[address][0][counter]) / (border - counter)
                    self.accounts[address][1][border] -= self.accounts[address][0][counter]
                    self.accounts[address][0].pop(counter)

        # Save only every Nth position of the array
        self.velocities[address] = (ind_velocity[::self.save_every_n] * self.save_every_n)

    def save_results(self):
        output_path = os.path.join(self.output_folder, 'general_velocities.pickle')
        with open(output_path, 'wb') as file:
            pickle.dump([self.accounts, self.velocities], file)

    def run_analysis(self):
        print("Loading allocated data...")
        self.load_allocated_data()
        print("Loading transfer data...")
        self.load_transfer_data()
        print(f"Min block number: {self.min_block_number}")
        print(f"Max block number: {self.max_block_number}")
        print(f"Number of blocks: {self.LIMIT}")
        print("Calculating velocities...")
        self.calculate_velocities()
        print("Saving results...")
        self.save_results()
        print(f"Min block number: {self.min_block_number}")
        print(f"Max block number: {self.max_block_number}")
        print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Micro Velocity Analyzer')
    parser.add_argument('--allocated_file', type=str, required=True, help='Path to the allocated CSV file')
    parser.add_argument('--transfers_file', type=str, required=True, help='Path to the transfers CSV file')
    parser.add_argument('--output_folder', type=str, default='temp', help='Path to the output folder')
    parser.add_argument('--save_every_n', type=int, default=1, help='Save every Nth position of the velocity array')
    args = parser.parse_args()

    analyzer = MicroVelocityAnalyzer(
        allocated_file=args.allocated_file,
        transfers_file=args.transfers_file,
        output_folder=args.output_folder,
        save_every_n=args.save_every_n
    )
    analyzer.run_analysis()

# Setup script for packaging
setup(
    name='micro_velocity_analyzer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    entry_points={
        'console_scripts': [
            'micro-velocity-analyzer=micro_velocity_analysis:main',
        ],
    },
    description='A package for analyzing account velocity based on allocated and transfer data.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/fdecollibus/micro_velocity_analyzer',
)
