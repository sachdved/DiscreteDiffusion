default_aa_keys='-GALMFWKQESPVICYHRNDT'

import torch
import numpy as np
import sklearn

import pandas as pd


def fasta_to_df(fasta_file, aa_keys = default_aa_keys):
    """
    creates one hot encoding of a fasta file using biopython's alignio.read process. 
    fasta_file : filepath leading to msa file in fasta format at hand
    """
    column_names = []
    column_names.extend(aa_keys)
    msa=AlignIO.read(fasta_file, "fasta")
    num_columns = len(msa[0].seq)
    column_names = column_names*num_columns
    column_names.append('sequence')
    column_names.append('id')
    init = np.zeros((len(msa), len(column_names)))
    df = pd.DataFrame(init, columns = column_names)
    df.sequence = df.sequence.astype(str)
    df.id=df.id.astype(str)
    
    for row_num, alignment in tqdm(enumerate(msa)):
        sequence = str(alignment.seq)
        for index, char in enumerate(sequence):
            place = aa_keys.find(char)
            df.iloc[row_num, index*len(aa_keys) + place] = 1
        
        df.iloc[row_num,-2]=str(alignment.seq)
        df.iloc[row_num,-1]=str(alignment.id)
    
    return df

def create_frequency_matrix(df, aa_keys = default_aa_keys):
    """takes one hot encoded msa and returns the frequency of each amino acid at each site
    df : pandas dataframe whose columns are the one hot encoding of an msa
    """
    num_columns=len(df['sequence'][0])
    
    frequency_matrix = np.zeros( (len(aa_keys) , num_columns) )
    print('calcing sum')
    freq=df.sum()
    print('sum calced')
    
    num_entries=len(df)
    len_aa_keys = len(aa_keys)
    
    for i in tqdm(range(len(aa_keys))):
        for j in range(num_columns):
            frequency_matrix[i, j] = freq[ i + len_aa_keys * j] / num_entries
    
    return frequency_matrix

if __name__ == '__main__':
    import torch
    import numpy as np
    import sklearn
    
    import pandas as pd

    msa = pd.read_csv('SH3_Full_Dataset_8_9_22.csv')
    msa['Type'].unique()
    naturals_msa = msa[msa['Type']=='Naturals']
    seqs = np.asarray([list(seq) for seq in naturals_msa['Sequences']])
    norm_re = np.asarray([re for re in naturals_msa['Norm_RE']])

    phyla = np.asarray([domain for domain in naturals_msa['Phylum']])

    from Bio import AlignIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from tqdm import tqdm
    vae_alignment = []
    phenotypes = []
    
    vae_data = msa[msa['Type']=='VAE'].reset_index()
    
    for r in range(len(vae_data)):
        alignment = vae_data.loc[r]
        if len(alignment['Sequences'])==62:
            record = SeqRecord(seq = Seq(alignment['Sequences']), id = alignment['Header'])
        
        vae_alignment.append(record)
        phenotypes.append(alignment['Norm_RE'])
    
    vae_alignment = AlignIO.MultipleSeqAlignment(vae_alignment)
    
    AlignIO.write(vae_alignment, 'vae_alignment.fasta', 'fasta')
    
    vae_df = fasta_to_df('vae_alignment.fasta')
    
    freq_matrix = create_frequency_matrix(vae_df)
    
    trim_positions = []
    
    for i in range(freq_matrix.shape[1]):
        if 1 in freq_matrix[:,i]:
            trim_positions.append(i)
    
    print(trim_positions)
    
    
    vae_alignment_trimmed = []
    
    
    for alignment in vae_alignment:
        new_seq = ''
        for i in range(62):
            if i not in trim_positions:
                new_seq+=alignment.seq[i]
        re_alignment = SeqRecord(seq=Seq(new_seq), id = alignment.id)
        vae_alignment_trimmed.append(re_alignment)
    
    vae_alignment_trimmed = AlignIO.MultipleSeqAlignment(vae_alignment_trimmed)
    
    AlignIO.write(vae_alignment_trimmed, 'vae_alignment_trimmed.fasta', 'fasta')
    
    test_seqs = np.asarray([list(str(alignment.seq)) for alignment in vae_alignment_trimmed])
    
    phenotypes = np.asarray(phenotypes)