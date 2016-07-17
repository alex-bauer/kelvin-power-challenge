import code,sys,gzip,time
import pandas as pd
import pickle

import logging

loggerInit = {}

def getLogger(name):

    global loggerInit

    logger = logging.getLogger(name)

    if not(name in loggerInit.keys()):
        loggerInit[name] = 1
        # get mac adress to have a logfile name
        from uuid import getnode as get_mac
        mac = get_mac()

        # make path for logfiles
        import os
        if not os.path.exists("../logs"):
            os.makedirs("../logs")

        # get name for the logfile
        logfilename = os.getcwd() + '/../logs/%i.log'%mac
        logfilename = os.path.normpath(logfilename)
        #print("Logfile name %s" % logfilename)
            
        # add a filehandler for the logfile
        fhandler = logging.FileHandler(filename=logfilename, mode='a')
        formatter = logging.Formatter('%(asctime)-15s %(name)-20s %(levelname)-8s %(message)s')
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.DEBUG)              

        # add console logger
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(asctime)-10s %(message)s', datefmt='%H:%M:%S',))
        logger.addHandler(console)
        
    # 
    #logger = getLogger("utils.py") 
    
    return logger
    
logger = getLogger("utils.py")
    
#def getLogger(name):

        
    #logging.getLogger('').addHandler(console)      
    
    #return logger 



#logger = logging.getLogger("utils.py")     

# matlab-style function for interactive debugging 
# just put keyboard() in the code to get interactive shell during execution
# from http://vjethava.blogspot.de/2010/11/matlabs-keyboard-command-in-python.html
def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print("# Use quit() to exit :) Happy debugging!")
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return 

# fancy progress printer
def progress_printer(current_number, total_number):
    """
    This function does nothing but displaying a fancy progress bar :)
    """
    
    global anim_state, pp_last_print_time
    if not 'anim_state' in globals():
        anim_state = 0
    
    # Chose if we have to print again?
    if 'pp_last_print_time' not in globals():
        printme = True
    elif time.time() - pp_last_print_time > 0.1:
        printme = True
    else:
        printme = False

    # What to print
    if printme:
        anim=["[*     ]","[ *    ]","[  *   ]","[   *  ]","[    * ]","[     *]","[    * ]","[   *  ]","[  *   ]","[ *    ]"]
        if total_number != None and total_number != 0:
            progress = str(int((float(current_number)/total_number)*100)) + "%"
        else:
            progress = " *working hard* (" + str("{:,}".format(current_number)) + " elements processed)"
        print("\r" + anim[anim_state] + " " + progress,end="",flush=True)
        anim_state = (anim_state + 1) % len(anim)
        pp_last_print_time = time.time()
        sys.stdout.flush()
   

# convert timestamp 
def convert_timestamp(df,index='ut_ms'):
    df[index] = pd.to_datetime(df[index], unit='ms')
    return df

def marsexpress_error(predictions,targets):
    diff = (targets - predictions) ** 2
    error = np.mean(diff.values) ** 0.5
    return error
    
# load data file, resample and dropnans
def load_csv(filename,interval=60,dropnan=True, do_convert_timestamp=True):
    print("loading %s"%filename,end="",flush=True)
    sys.stdout.flush()
    df = pd.read_csv(filename)
    timestamp_col = 'ut_ms'
    if timestamp_col not in df.columns:
        # for ftl
        timestamp_col = 'utb_ms'
    if do_convert_timestamp:
        df = convert_timestamp(df,index=timestamp_col)
        df = df.set_index(timestamp_col)
        
    if interval>0:
        df = df.resample('%dT'%interval).mean()
    if dropnan:
        df = df.dropna()
    print("done")
    return df

def save_to_pickle(filename,df):
    print("saving to %s"%filename)
    f = gzip.open(filename,'wb')
    pickle.dump(df,f)
    f.close()
    
def load_raw_data(filenames_train,filename_test,interval=60):
    df = pd.DataFrame()
    # load train data
    for filename in filenames_train:
        df = pd.concat([df, load_csv(filename,interval)])
    # load test data
    df = pd.concat([df, load_csv(filename_test,interval=interval,dropnan=False)])
    return df
    
def load_prepare_pickle(name,filenames_train,filename_test,interval=60):
    """
    prepares the data and saves it to pickle file
    good for
        power (targets)
        saaf
        ltdata
    """
    df = load_raw_data(filenames_train,filename_test,interval)
    filename = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)
    save_to_pickle(filename,df)
    return df
	
# ---------------------------------------------------------------------------------------------------------------
    
def prepare_saaf(interval=60):
    filenames_train = ["../data/train_set/context--2008-08-22_2010-07-10--saaf.csv", \
                 "../data/train_set/context--2010-07-10_2012-05-27--saaf.csv", \
                 "../data/train_set/context--2012-05-27_2014-04-14--saaf.csv"
                ]
    filename_test = "../data/test_set/context--2014-04-14_2016-03-01--saaf.csv"
    name = "saaf"
    df = load_prepare_pickle(name,filenames_train,filename_test,interval)
    return df
    
# ---------------------------------------------------------------------------------------------------------------
	
def prepare_power(interval=60):
    filenames_train = ["../data/train_set/power--2008-08-22_2010-07-10.csv", \
                     "../data/train_set/power--2010-07-10_2012-05-27.csv", \
                     "../data/train_set/power--2012-05-27_2014-04-14.csv"
                    ]
    filename_test = "../data/power-prediction-sample-2014-04-14_2016-03-01.csv"
    name = "power"
    df = load_prepare_pickle(name,filenames_train,filename_test,interval)
    return df
	
 # ---------------------------------------------------------------------------------------------------------------   
 
def prepare_ltdata(interval=60):
    filenames_train = ["../data/train_set/context--2008-08-22_2010-07-10--ltdata.csv", \
                 "../data/train_set/context--2010-07-10_2012-05-27--ltdata.csv", \
                 "../data/train_set/context--2012-05-27_2014-04-14--ltdata.csv"
                ]
    filename_test = "../data/test_set/context--2014-04-14_2016-03-01--ltdata.csv"
    name = "ltdata"
    load_prepare_pickle(name,filenames_train,filename_test,interval)

# ---------------------------------------------------------------------------------------------------------------    

def prepare_evtf(interval=60, flag_save_to_pickle=True):
    filenames_train = ["../data/train_set/context--2008-08-22_2010-07-10--evtf.csv", \
                 "../data/train_set/context--2010-07-10_2012-05-27--evtf.csv", \
                 "../data/train_set/context--2012-05-27_2014-04-14--evtf.csv"
                ]
    filename_test = "../data/test_set/context--2014-04-14_2016-03-01--evtf.csv"
    name = "evtf"
    
    df = load_raw_data(filenames_train,filename_test,0)
    
    
    # interval=0 means : save raw data
    if interval!=0:
        # first shot: only import umras
        umbras = pd.get_dummies(df.loc[df.description.str.contains("^MAR.*UMBRA", regex=True),:])
        df = pd.DataFrame(index=df.index)
        df = df.join(umbras).fillna(0)
        # TODO this way of resampling is crude!
        df = df.resample('%dT'%interval).max().fillna(0)
    # save to pickle file
    if flag_save_to_pickle:
        filename = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)    
        save_to_pickle(filename,df)
        
    return df
	
def getVarValuePair(x):
    
    # remove the "/" in the MRB_/_RANGE split to make it a variable
    desc = x.description.replace("_/_RANGE","_RANGE")
        
    # split off all info that is auxiliary
    descArr = desc.split("/")
    varName = descArr[0]
    
    # "PERICENTRE_PASSAGE_05956" needs to be cleaned up
    if "CENTRE_PASSAGE" in varName:
        varNameArr = varName.split("_")
        varName = varNameArr[0]+varNameArr[1] # we remove the number in the end
    
    # we are looking for the following values
    valueKeys = ["DESCEND", "ASCEND", "START", "END", "AOS", "LOS", "APO", "PERI"] # workaround because dict.keys() looses order
    values = {"DESCEND" : 1,
              "ASCEND" : 0,
              "START": 1,
              "END": 0,
              "AOS": 1,
              "LOS": 0,
              "APO": 1,
              "PERI": 0}
    value = "???"
    
    # see if we can find it in the variable name and split 
    for val in valueKeys: # values.keys(): does not work (does not preserver the order)
        if val in varName:
            varName = varName.replace(val,"")
            value = values[val]
            break
    
    varName=varName.strip("_")
    
    return (varName, value)

def getVarName(x):
    
    return getVarValuePair(x)[0]

def getValue(x):
    
    return getVarValuePair(x)[1]


def prepare_evtf2():
    
    evtf = prepare_evtf(0,False)

    print("Converting to Variables / Values")
    evtf["ZZ_VarName"] = evtf.apply(getVarName, axis=1)
    evtf["ZZ_Value"] = evtf.apply(getValue, axis=1)
    
    print("Converting to columns")
    df = evtf.reset_index().pivot(columns='ZZ_VarName')['ZZ_Value'] #index='label',
    
    print("Adding timestamp back in ")
    df2 = pd.concat([ evtf.reset_index().loc[:,["ut_ms", "description"]], df], axis=1)
    df2 = df2.set_index("ut_ms")
    
    # filling the na variables
    df3 = df2.fillna(method="ffill")
    
    # challenge: how to fill the still existing "na"?
    # option1: We could assume that the value flips when it is present the first time
    # option2: we fill it with 0 and hope it's the right choice (it might be, because we do not now it is "on")
    # for now, we will take option 2
    # those before the first occurence will be filled with 0
    df4 = df3.fillna(0)

    # before resampling, we need to fix the duplicates
    df5 = df4.reset_index()
    import time
    import datetime
    df5["ts"] = df5["ut_ms"].apply(lambda x: int(time.mktime(x.timetuple())*1000)) 
   
    # idea: We add +1 to all the timestamps that are duplicates
    df6 = df5["ts"].shift()
    duplicates = (df5["ts"]-df6)==0    
    df5.loc[duplicates,"ts"] += 1
    
    # and convert it back to the timestamp
    df6 = convert_timestamp(df5, "ts")
    df6 = df6.set_index("ts")
    
    # resample to 1 minute
    #df7 = df6.resample("1T").ffill()
    
    # and now upsample to 60min with sum to get the minutes per hour
    #df8 = df7.fillna(method="bfill").resample("60T").sum()
    df8 = df6
    
    name="evtf-features2"
    interval = 0
    
    filename = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)    
    save_to_pickle(filename,df8)
    
    return df8
  
# ---------------------------------------------------------------------------------------------------------------  
    
def prepare_dmop(interval=60, flag_save_to_pickle=True):
    filenames_train = ["../data/train_set/context--2008-08-22_2010-07-10--dmop.csv", \
                 "../data/train_set/context--2010-07-10_2012-05-27--dmop.csv", \
                 "../data/train_set/context--2012-05-27_2014-04-14--dmop.csv"
                ]
    filename_test = "../data/test_set/context--2014-04-14_2016-03-01--dmop.csv"
    name = "dmop"
    
    df = load_raw_data(filenames_train,filename_test,0)
    
    
    if interval!=0:
        # strip command number 
        df.subsystem = df.subsystem.apply(lambda d: str.split(d,".")[0])
        df = df.join(pd.get_dummies(df))
        df = df.drop(["subsystem"], axis=1)
        # TODO this way of resampling is crude!
        df = df.resample('%dT'%interval).max().fillna(0)
        
    # save to pickle file
    if flag_save_to_pickle:
        filename = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)
        save_to_pickle(filename,df)
        
    return df

def prepare_ftl(interval=60, flag_save_to_pickle=True):
    filenames_train = ["../data/train_set/context--2008-08-22_2010-07-10--ftl.csv", \
                 "../data/train_set/context--2010-07-10_2012-05-27--ftl.csv", \
                 "../data/train_set/context--2012-05-27_2014-04-14--ftl.csv"
                ]
    filename_test = "../data/test_set/context--2014-04-14_2016-03-01--ftl.csv"
    name = "ftl"
    
    df = load_raw_data(filenames_train,filename_test,0)
    if interval!=0:
        # hard coded intervals from data (first and last time stamps)
        timerange = pd.to_datetime([1219363200000,1456786800000], unit='ms')
        myindex = pd.date_range(start=timerange[0], end=timerange[1], freq='%dT'%interval)
        
        df2 = pd.DataFrame(index=myindex)
        
        colList = ['flagcomms',"ACROSS_TRACK","D1PVMC","D2PLND","D3POCM","D4PNPO","D5PPHB",\
        "D7PLTS","D8PLTP","D9PSPO","EARTH","INERTIAL","MAINTENANCE",\
        "NADIR","RADIO_SCIENCE","SLEW","SPECULAR","SPOT","WARMUP","NADIR_LANDER",]        
        
        for col in colList:
            df2[col] = 0
            if col != 'flagcoms':
                df2[col+"-started"] = 0
                df2[col+"-ended"] = 0
                df2[col+"-cumulativeActiveMinutes"] = 0

        # format times in data
        df['ute_ms'] = pd.to_datetime(df['ute_ms'], unit='ms')
        for i in range(len(df)):
            
            if ((i+1) % 100) == 0:
                progress_printer(i,len(df))
            # get masks indicating when event type is active
            #begin = df2.index > df.index[i]
            #end = df2.index < df.ute_ms[i]
            # get first bin in which event type is active 
            firstindex = df2.index.searchsorted(df.index[i])-1
            lastindex = df2.index.searchsorted(df.ute_ms[i])-1
            # set the bin with where the event begins to 1
            #TODO this could be more precise than 0/1
            
            
            if lastindex == firstindex:
                # calculate time for this event 
                active_time_in_first_bin = (df.ute_ms[i]-df.index[i]).components.minutes
            elif lastindex > firstindex:
                # calculate number of minutes for event in first bin
                active_time_in_first_bin = (df2.index[firstindex+1]-df.index[i]).components.minutes
                # calculate number of minutes for event in last bin
                active_time_in_last_bin = (df.ute_ms[i]-df2.index[lastindex]).components.minutes
                # set feature to be fully active in all bins within the range
                df2[df.type[i]][firstindex+1:lastindex] = interval
                # write active time to last bin
                df2[df.type[i]][lastindex] = max(df2[df.type[i]][lastindex],active_time_in_last_bin)
                # same for flagcomms
                if df.flagcomms[i]:
                    df2.flagcomms[firstindex+1:lastindex-1] = interval
                    df2.flagcomms[lastindex] = max(df2.flagcomms[lastindex],active_time_in_last_bin)
            
            # first bin: add possible event duration of previous event type in same bin
            total_active_time_in_first_bin = min(interval,df2[df.type[i]][firstindex]+active_time_in_first_bin)
                
            # write active time in first bin. assumption: non-overlapping times of same features
            df2[df.type[i]][firstindex] = total_active_time_in_first_bin
            
            # TODO this does not catch all special cases when same event type occurs again in this bin (in later data point)
            if total_active_time_in_first_bin == active_time_in_first_bin:
                # set flag in bin where event type started
                df2[df.type[i]+"-started"][firstindex] = 1
            
            # set flag in bin where event type ended
            df2[df.type[i]+"-ended"][lastindex] = 1
            # and we're pretty sure it didnt end in the first bin
            df2[df.type[i]+"-ended"][firstindex] = 0
            
            # calculate cumulative times
            if lastindex > firstindex:
                # cumulative minutes from this event activation
                cumulativeActiveMinutes = \
                    total_active_time_in_first_bin + interval*np.arange(0,lastindex-(firstindex))
                # add possible previous time
                cumulativeActiveMinutes += \
                        df2[df.type[i]+"-cumulativeActiveMinutes"][firstindex]
                df2[df.type[i]+"-cumulativeActiveMinutes"][firstindex+1:lastindex+1] = cumulativeActiveMinutes
            # same for flagcomms
            if df.flagcomms[i]:
                df2.flagcomms[firstindex] = min(interval,df2.flagcomms[firstindex]+active_time_in_first_bin)
            
        # use dummies, resampled to interval
        df = df2
    # save to pickle file
    if flag_save_to_pickle:    
        filename = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)
        save_to_pickle(filename,df)
        
    return df
    
# ------------------------------------------------------- added by SeAp


def loadAllData(resamplerate):
    
    logger.info("loading all data")
    
    interval = resamplerate
    
    name = "power"
    fname = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)
    power = pickle.load( gzip.open( fname, "rb" ) )
    
    data_types = ["saaf","ltdata","ftl","evtf","dmop"]
    data = pd.DataFrame(index=power.index)
    for name in data_types:
        logger.debug ("loading pickled %s data"%name)
        fname = "../data/preprocessed/%s-%dmin.pklz"%(name,interval)    
        df_tmp = pickle.load( gzip.open( fname, "rb" ) )
        # Careful here: we need to make sure we return indices that are the same for X and Y
        df_tmp = df_tmp.reindex(power.index, method='nearest').interpolate()
        data = data.join(df_tmp)
    return (data, power) 

    

import numpy as np
# Defining the evaluation metric
def RMSE(val, pred):
    diff = (val - pred) ** 2
    rmse = np.mean(diff.values) ** 0.5
    return rmse
