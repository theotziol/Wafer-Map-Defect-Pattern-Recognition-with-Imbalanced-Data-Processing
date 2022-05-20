import numpy as np
import PIL
from PIL import Image
from copy import deepcopy as dc
import pandas as pd
def resize(x, size = 128, rgb = False):
    x = Image.fromarray(x)
    new_x = np.array(x.resize((size,size), resample = PIL.Image.NEAREST))
    new_x = np.reshape(new_x, (size,size,1))
    if rgb:
        new_x = np.stack((new_x,)*3, axis = -1)
    return new_x


def augment(df, factor = 3):
    '''Factor 1, 2 or 3'''
    l = len(df)
    y = df['failureNum']
    flip_h = df['waferMap'].apply(np.flip,args = (1,))
    flip_v = df['waferMap'].apply(np.flip,args = (0,))
    rot90 = df['waferMap'].apply(np.rot90)
    rot180 = rot90.apply(np.rot90)
    rot270 = rot180.apply(np.rot90)
    flip_rot_h = flip_h.apply(np.rot90)
    flip_rot_v = flip_v.apply(np.rot90)

    if factor == 1:
        y2 = pd.concat([y,y], ignore_index=True)
        aug = pd.concat([flip_h,flip_v], ignore_index=True)
        df2 = pd.concat([aug,y2], axis = 1)
        final_df = pd.concat([df,df2], ignore_index=True)
        print('Dataframe old len {}, new datarame len {}'.format(l,len(final_df)))
        return final_df

    elif factor == 2:
        y2 = pd.concat([y,y,y,y], ignore_index=True)
        aug = pd.concat([flip_h,flip_v,rot90,rot180], ignore_index=True)
        df2 = pd.concat([aug,y2], axis = 1)
        final_df = pd.concat([df,df2], ignore_index=True)
        print('Dataframe old len {}, new datarame len {}'.format(l,len(final_df)))
        return final_df
    else:
        y2 = pd.concat([y,y,y,y,y,y,y], ignore_index=True)
        aug = pd.concat([flip_h,flip_v,rot90,rot180, rot270, flip_rot_h, flip_rot_v], ignore_index=True)
        df2 = pd.concat([aug,y2], axis = 1)
        final_df = pd.concat([df,df2], ignore_index=True)
        print('Dataframe old len {}, new datarame len {}'.format(l,len(final_df)))
        return final_df


def split(df, t = 0.7, shuffle = True):
    n = len(df)
    if shuffle:
        df = dc(df.sample(frac = 1))
    tr = df.iloc[:int(t*n)]
    ts = df.iloc[int(t*n):]
    return tr,ts

def downsample(df, samples):
    df1 = dc(df.sample(frac = 1))
    return df1.iloc[:samples]
    
