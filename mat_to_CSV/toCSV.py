import scipy.io
import pandas as pd
import os

def toCSV(subject_num, source):
    '''
    converts the signal data from the .mat files in the as_mat subdirectory to csv files in as_csv
    @param subject_num : str
        the number of the subject whose data we want to convert
    @param source : int
        indicates how a mat file should have the signal data extracted
        (which workspace variable stores the data)
    '''
    if not source:
        print("input a source!")
        return
    directory = os.fsencode('as_mat')
    file_index = 1
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mat"):
            print(f"now converting 'as_mat\\{filename}'")
            mat_data = scipy.io.loadmat(f'as_mat\\{filename}')
            
            match source:
            # expand with more cases as needed
                case 1:
                    signal_data = mat_data['signal']
                case 2:
                    signal_data = mat_data['dataRest']
            
            df = pd.DataFrame(signal_data)
            # update the subject number below as needed
            df.to_csv(f'as_csv\\Sub_{subject_num}_Block_{file_index}.csv')
            file_index+=1
            continue
        else:
            continue
    print("all mat files converted")
    pass

def transposeCSV(csv_table_name, endInd):
    '''
    transposes the rows and columns of a given table and saves it
    optionally slice the transposed table past the end index
    @param csv_table_name : str
        name of a csv table located in the as_csv directory
    @param endInd : int
        positive int indicating the number of columns from the right the dataset should be
    '''
    df = pd.read_csv(f'as_csv\\{csv_table_name}')
    df_transposed = df.transpose()

    if (isInstance(endInd, int)):
        df_transposed = crop(df_transposed, endInd)

    df_transposed.to_csv(f'as_csv\\transposed{csv_table_name[:-4]}.csv')
    pass

def crop(dataframe, endInd):
    return dataframe.iloc[:,:endInd]

def deleteMats():
    directory = os.fsencode('as_mat')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".mat"):
            if os.path.exists(f'as_mat\\{filename}'):
                print(f"deleting {filename}!")
                os.remove(f'as_mat\\{filename}')
            else:
                print("The file does not exist")
                continue
        else:
            continue
    pass

def main():
    print('starting')
    # toCSV('1', 1)
    # deleteMats()
    print('finished')
    return

if __name__ == '__main__':
    main()




