import pandas as pd
from pathlib import Path
import numpy as np


def read_sim_summary(output_summary_path, mm_summary_path):

    col_names = ['scenario', 'mean_waiti', 'mean_waitp', 'mean_time_in_system', 'mean_eod']
    summary_stats_df = pd.read_csv(output_summary_path, names=col_names)

    summary_stats_df.to_csv(mm_summary_path, index=False)
    return summary_stats_df

def offered_util(row, resource):
    if resource == 'staff':
        numerator = \
            (row.vitals_time_mean + row['post_exam_time_mean'] + row['room_turnover_time_mean']) * row['patients_per_clinic_block']
        denom = row['clinic_length_minutes'] * row['num_med_techs']
    elif resource == 'physician':
        numerator = \
            row['exam_time_mean'] * row['patients_per_clinic_block']
        denom = row['clinic_length_minutes'] * row['num_physicians']
    elif resource == 'room':
        numerator = \
            (row['post_exam_time_mean'] +
             row['room_turnover_time_mean'] + row['exam_time_mean']) * row['patients_per_clinic_block']
        denom = row['clinic_length_minutes'] * row['num_physicians'] * row['num_rooms_per_provider']
    else:
        numerator = 0
        denom = 1

    return numerator / denom



def make_xy_files(input_scenarios_filepath, mm_summary_filepath, xy_path):

    scenarios_df = pd.read_csv(input_scenarios_filepath)
    scenarios_df['exam_time_cv'] = 1 / np.sqrt(scenarios_df['exam_time_k'])
    scenarios_df['off_util_staff'] = scenarios_df.apply(lambda x: offered_util(x, 'staff'), axis=1)
    scenarios_df['off_util_physician'] = scenarios_df.apply(lambda x: offered_util(x, 'physician'), axis=1)
    scenarios_df['off_util_room'] = scenarios_df.apply(lambda x: offered_util(x, 'room'), axis=1)

    summary_stats_df = pd.read_csv(mm_summary_filepath)

    full_df = scenarios_df.merge(summary_stats_df, on='scenario')

    x_no_util_cols = ['patients_per_clinic_block', 'num_med_techs', 'num_rooms_per_provider',
                      'vitals_time_mean', 'exam_time_mean', 'exam_time_cv', 'post_exam_time_mean']

    x_util_cols = ['patients_per_clinic_block', 'num_med_techs', 'num_rooms_per_provider',
                   'vitals_time_mean', 'exam_time_mean', 'exam_time_cv', 'post_exam_time_mean',
                   'off_util_staff', 'off_util_physician', 'off_util_room']

    xy_no_util_cols_waiti = x_no_util_cols + ['mean_waiti']
    xy_no_util_cols_waitp = x_no_util_cols + ['mean_waitp']
    xy_no_util_cols_atic = x_no_util_cols + ['mean_time_in_system']
    xy_no_util_cols_eod = x_no_util_cols + ['mean_eod']

    xy_util_cols_waiti = x_util_cols + ['mean_waiti']
    xy_util_cols_waitp = x_util_cols + ['mean_waitp']
    xy_util_cols_atic = x_util_cols + ['mean_time_in_system']
    xy_util_cols_eod = x_util_cols + ['mean_eod']

    xy_no_util_waiti_df = full_df[xy_no_util_cols_waiti]
    xy_no_util_waitp_df = full_df[xy_no_util_cols_waitp]
    xy_no_util_atic_df = full_df[xy_no_util_cols_atic]
    xy_no_util_eod_df = full_df[xy_no_util_cols_eod]

    xy_util_waiti_df = full_df[xy_util_cols_waiti]
    xy_util_waitp_df = full_df[xy_util_cols_waitp]
    xy_util_atic_df = full_df[xy_util_cols_atic]
    xy_util_eod_df = full_df[xy_util_cols_eod]

    out_filepath = Path(xy_path) / Path('xy_no_util_waiti.csv')
    xy_no_util_waiti_df.to_csv(out_filepath, index=False)

    out_filepath = Path(xy_path) / Path('xy_no_util_waitp.csv')
    xy_no_util_waitp_df.to_csv(out_filepath, index=False)

    out_filepath = Path(xy_path) / Path('xy_no_util_atic.csv')
    xy_no_util_atic_df.to_csv(out_filepath, index=False)

    out_filepath = Path(xy_path) / Path('xy_no_util_eod.csv')
    xy_no_util_eod_df.to_csv(out_filepath, index=False)
    
    # with util terms
    out_filepath = Path(xy_path) / Path('xy_util_waiti.csv')
    xy_util_waiti_df.to_csv(out_filepath, index=False)

    out_filepath = Path(xy_path) / Path('xy_util_waitp.csv')
    xy_util_waitp_df.to_csv(out_filepath, index=False)

    out_filepath = Path(xy_path) / Path('xy_util_atic.csv')
    xy_util_atic_df.to_csv(out_filepath, index=False)

    out_filepath = Path(xy_path) / Path('xy_util_eod.csv')
    xy_util_eod_df.to_csv(out_filepath, index=False)


if __name__ == '__main__':

    output_summary_filepath = Path('output/summary_stats.csv')
    mm_summary_filepath = Path("mm_input/summary_stats.csv")
    xy_path = Path('./mm_input')
    summary_stats_df = read_sim_summary(output_summary_filepath, mm_summary_filepath)

    input_scenarios_filepath = Path('input/exp11_scenarios.csv')

    make_xy_files(input_scenarios_filepath, mm_summary_filepath, xy_path)
