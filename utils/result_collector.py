import logging
from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


class ResultCollector():

    def __init__(self, dataset_name, experiment_type):
        self.logger = logging.getLogger(__name__)

        self.results = {}
        self.dataset_name = dataset_name.split('.')[0]
        self.experiment_type = experiment_type

    def persist_results(self, timestamp):
        """Persist Experiment Results"""
        project_dir = Path(f'{self.experiment_type}')
        relative_path = project_dir.joinpath('results')
        
        Path(relative_path).mkdir(parents=True, exist_ok=True)
        absolute_path = Path(relative_path)

        file_path = absolute_path.joinpath('{}_evaluation-results_{}.csv'.format(
                            self.dataset_name,
                            datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M')))

        header = ['Experiment Name','Dataset','Split']
        # Use first experiment as reference for the metric header
        metric_header = list(list(self.results.values())[0].keys())
        header = header + metric_header

        rows = []
        for result in self.results.keys():
            if '+' in result:
                result_parts = result.split('+')
                experiment_name = result_parts[0]
                split = result_parts[1]
                row = [experiment_name, self.dataset_name, split]
            else:
                row = [result, self.dataset_name, 'all']
            for metric in self.results[result].items():
                row.append(metric[1])
            rows.append(row)

        # Write to csv
        with open(file_path, 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=';')

            csv_writer.writerow(header)
            csv_writer.writerows(rows)

        pd.read_csv(file_path, header=None, delimiter=';', encoding = "latin").T.to_csv(file_path, header=False, index=False, sep=';')

        self.logger.info('Results of {} on {} written to file {}!'.format(
                            self.experiment_type, self.dataset_name, file_path.absolute()))