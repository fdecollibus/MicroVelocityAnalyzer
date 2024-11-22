import argparse
import os
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, when, lit, lower, lag
from pyspark.sql.window import Window
from pyspark.sql.types import DecimalType, LongType, StringType
from decimal import Decimal

class MicroVelocityAnalyzer:
    def __init__(self, allocated_file, transfers_file, output_file='temp/general_velocities.pickle', save_every_n=1):
        self.allocated_file = allocated_file
        self.transfers_file = transfers_file
        self.output_file = output_file
        self.save_every_n = save_every_n
        self._create_output_folder()
        self.spark = SparkSession.builder.appName("MicroVelocityAnalyzer").getOrCreate()
        self.spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    def _create_output_folder(self):
        output_folder = os.path.dirname(self.output_file)
        if output_folder and not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def load_data(self):
        print("Loading allocated data...")
        allocated_df = self.spark.read.csv(
            self.allocated_file,
            header=True,
            inferSchema=False
        ).select(
            lower(col('to_address')).alias('address').cast(StringType()),
            col('amount').cast(StringType()),
            col('block_number').cast(LongType())
        ).withColumn('type', lit('allocation'))

        print("Loading transfer data...")
        transfers_df = self.spark.read.csv(
            self.transfers_file,
            header=True,
            inferSchema=False
        )

        transfers_in = transfers_df.select(
            lower(col('to_address')).alias('address').cast(StringType()),
            col('amount').cast(StringType()),
            col('block_number').cast(LongType())
        ).withColumn('type', lit('transfer_in'))

        transfers_out = transfers_df.select(
            lower(col('from_address')).alias('address').cast(StringType()),
            (-col('amount').cast(StringType())).alias('amount'),
            col('block_number').cast(LongType())
        ).withColumn('type', lit('transfer_out'))

        self.all_transactions = allocated_df.union(transfers_in).union(transfers_out)

        # Convert amount to Decimal with high precision
        decimal_type = DecimalType(38, 0)  # Adjust precision and scale as needed
        self.all_transactions = self.all_transactions.withColumn('amount', col('amount').cast(decimal_type))

        # Compute min and max block numbers
        self.min_block_number = self.all_transactions.agg({"block_number": "min"}).collect()[0][0]
        self.max_block_number = self.all_transactions.agg({"block_number": "max"}).collect()[0][0]
        self.LIMIT = self.max_block_number - self.min_block_number + 1
        print(f"Min block number: {self.min_block_number}")
        print(f"Max block number: {self.max_block_number}")
        print(f"Number of blocks considered: {self.LIMIT}")

    def calculate_balances_and_velocities(self):
        print("Calculating balances and velocities...")

        # Group transactions by address and block_number and sum amounts
        grouped = self.all_transactions.groupBy('address', 'block_number').agg(
            spark_sum('amount').alias('net_amount')
        )

        # Define window for cumulative sum
        window_spec = Window.partitionBy('address').orderBy('block_number').rowsBetween(Window.unboundedPreceding, 0)

        # Calculate cumulative balance per address
        balances = grouped.withColumn('balance', spark_sum('net_amount').over(window_spec))

        # Calculate velocities
        window_spec_lag = Window.partitionBy('address').orderBy('block_number')
        balances = balances.withColumn('prev_balance', lag('balance', 1).over(window_spec_lag))
        balances = balances.withColumn('prev_block_number', lag('block_number', 1).over(window_spec_lag))

        balances = balances.withColumn('delta_balance', col('balance') - col('prev_balance'))
        balances = balances.withColumn('delta_time', col('block_number') - col('prev_block_number'))
        balances = balances.withColumn('velocity', when(col('delta_time') > 0, col('delta_balance') / col('delta_time')).otherwise(0))

        # Fill null values
        balances = balances.fillna({'velocity': 0, 'delta_time': 1})

        # Save every Nth position
        if self.save_every_n > 1:
            balances = balances.filter((col('block_number') - self.min_block_number) % self.save_every_n == 0)

        # Collect results
        self.balances = balances.select('address', 'block_number', 'balance').collect()
        self.velocities = balances.select('address', 'block_number', 'velocity').collect()

    def save_results(self):
        print("Saving results...")
        # Convert collected rows to dictionaries
        balances_dict = {}
        for row in self.balances:
            address = row['address']
            if address not in balances_dict:
                balances_dict[address] = []
            balances_dict[address].append((row['block_number'], row['balance']))

        velocities_dict = {}
        for row in self.velocities:
            address = row['address']
            if address not in velocities_dict:
                velocities_dict[address] = []
            velocities_dict[address].append((row['block_number'], row['velocity']))

        with open(self.output_file, 'wb') as file:
            pickle.dump({'balances': balances_dict, 'velocities': velocities_dict}, file)

    def run_analysis(self):
        self.load_data()
        self.calculate_balances_and_velocities()
        self.save_results()
        print("Done!")
        self.spark.stop()

def main():
    parser = argparse.ArgumentParser(description='Micro Velocity Analyzer')
    parser.add_argument('--allocated_file', type=str, default='sampledata/sample_allocated.csv', help='Path to the allocated CSV file')
    parser.add_argument('--transfers_file', type=str, default='sampledata/sample_transfers.csv', help='Path to the transfers CSV file')
    parser.add_argument('--output_file', type=str, default='sampledata/general_velocities.pickle', help='Path to the output file')
    parser.add_argument('--save_every_n', type=int, default=1, help='Save every Nth position of the velocity array')
    args = parser.parse_args()

    analyzer = MicroVelocityAnalyzer(
        allocated_file=args.allocated_file,
        transfers_file=args.transfers_file,
        output_file=args.output_file,
        save_every_n=args.save_every_n
    )
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
