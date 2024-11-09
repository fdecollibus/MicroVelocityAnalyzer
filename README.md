# MicroVelocityAnalyzer

MicroVelocityAnalyzer is a Python package for analyzing the velocity and balances of accounts based on transfer and allocation data.
This package is based on the initial inspiration of Carlo Campajola, a inspiration that has been explored in many papers about different cryptocurrencies. This package is based on the work made on Ethereum MicroVelocity on an upcoming paper, and is developed and mantained by Francesco Maria De Collibus and Carlo Campajola

## Features

- Load allocation data from a CSV file.
- Load transfer data from a CSV file.
- Calculate account velocities.
- Calculate account balances.
- Save results to a pickle file.

## Installation

To install the package, run the following command:

```sh
pip install -e .
```

## Usage

After installing the package, you can run the script from the command line using the `micro-velocity-analyzer` command. Here is an example usage:

```sh
micro-velocity-analyzer --allocated_file path/to/allocated.csv --transfers_file path/to/transfers.csv --output_file path/to/output/general_velocities.pickle --save_every_n 10
```

## Caution

To save space, velocities and balances are sampled according to "save_every_n" parameter. In the saved file with results, the first element are accounts with assets and liabilities, the second element are velocities, and the third are balances. Hence you can use them to calculate everything you need.

### Arguments

- `--allocated_file`: Path to the CSV file containing allocation data (required).
- `--transfers_file`: Path to the CSV file containing transfer data (required).
- `--output_file`: Path to the output file where results will be saved (optional, default: `temp/general_velocities.pickle`).
- `--save_every_n`: Save every Nth position of the velocity array (optional, default: 1).

## Example CSV Files

### allocated.csv

```csv
to_address,amount,block_number
address1,100,1
address2,200,2
address1,150,3
```

### transfers.csv

```csv
from_address,to_address,amount,block_number
address1,address2,50,4
address2,address3,100,5
address1,address3,75,6
```

## Contributions

Contributions are welcome! Feel free to open issues or pull requests to improve the project.

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

## Author

Francesco Maria De Collibus - [francesco.decollibus@business.uzh.ch](mailto:francesco.decollibus@business.uzh.ch)

```

```

```

```
