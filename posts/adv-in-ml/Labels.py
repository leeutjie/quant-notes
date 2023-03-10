import numpy as np
import pandas as pd


class ClassificationLabels:
    """
    Methods associated with classification labelling in financial machine learning models.
    Adapted from Advances in Machine Learning by Marcos Lopez de Prado.

    Attributes:
        S: A series of prices 
        log_S: A series log prices
    """
    def __init__(self, S):
        """
        Args:
            S: A series of prices
        """
        self.S = S
        self.log_S = np.log(S)

    def cusum_filter_sym(self, s:float)->pd.DatetimeIndex:
        """
        Samples events based on the symmetric cusum filter.

        Args: 
            s: A threshold (logarithmic scale).

        Returns: 
            events: A list of datetime indices.
        """
        events, s_pos, s_neg = [self.log_S.index[0]],0,0
        diff = self.log_S.diff()

        # Sum over positive and negative change respectively.
        for i in diff.index[1:]:
            s_pos, s_neg = max(0,s_pos+diff.loc[i]), min(0,s_neg+diff.loc[i])
            if s_neg < -s:
                s_neg=0
                events.append(i)
            elif s_pos > s:
                s_pos=0
                events.append(i)

        events = pd.DatetimeIndex(events)
        return events


    def cusum_filter_asym(self, s:float)->pd.DatetimeIndex:
        """
        Samples events based on the asymmetric cusum filter.

        Args:
            s - A threshold (logarithmic scale).

        Returns:
            events: A list of datetime indices.
        """
        events,s_pos = [self.log_S.index[0]],0
        diff = self.log_S.diff()

        # Sum over positive change only.
        for i in diff.index[1:]:
            s_pos = max(0, s_pos+diff.loc[i])
            if s_pos > s:
                s_pos=0
                events.append(i)

        events = pd.DatetimeIndex(events)
        return events


    def cusum_interp(self, events:list)->pd.Series:
        """
        Finds the linearly interpolated values of the sampled events corresponding to the indices of the full dataset.

        Args:
            events: A list of datetime indices.

        Returns:
            events_interp: A series of interpolated events based on the sampled events.
        """
        events_loc = self.S[events]
        S_o = pd.Series(events_loc, index=self.S.index)
        S_interp = S_o.interpolate()
        return S_interp
        

    def abs_return_forward(self, h:int)->pd.Series:
        """
        Forward return.
        Calculates the absolute price return over a period h. 

        Args:
            h: The horizon, in days, over which the return is calculated.

        Returns: 
            r: A series of absolute price returns.
        """
        fwd_dates = []  

        for date,price in self.S.items():
            fwd_date = date+pd.Timedelta(days=h)    # Forward date.
            diff = (self.S.index - fwd_date).days.values    # Find the difference between the forward date and the later dates in the proce series.
            diff = np.ma.MaskedArray(diff, diff<0)  # We are not interested in the dates before the forward date.
            if diff.mask.all(): 
                fwd_date = pd.NaT   # Occurs when there are no dates to choose from.    
            else: 
                fwd_date = self.S.index.values[diff.argmin()] # Finds the earliest corresponding date in the price series.
            fwd_dates.append(fwd_date)
      
        S_o = pd.Series(fwd_dates, index=self.S.index).dropna()
        r = (self.S.loc[S_o.values]/self.S.loc[S_o.index.values].values-1) # Forward return.
        r= pd.Series(r.values, S_o.index)
      
        return r
       
    def abs_return_forward_2(self, h:int)->pd.Series:
        """
        Forward return version 2.
        Calculates the absolute price return over a period h. 

        Args:
            h - The horizon, in days, over which the return is calculated.

        Returns:
            r: A series of absolute price returns.
        """
        idx = self.S.index.searchsorted(self.S.index+pd.Timedelta(days=h)) # Get the index positions. This will create more index positions than the number in the price series.
        idx = idx[idx<self.S.shape[0]] # Take only the index values that would exist in the price series.
        S_o = pd.Series(self.S.index[idx], index=self.S.index[:len(idx)]) # Whats left is a series with less values than originally in the price series. 
                                                                          # Since we are looking at the forward difference we will be missing returns corresponding to the end of the price series.
        r = self.S.loc[S_o.values]/self.S.loc[S_o.index.values].values-1 # Forward return.
        r= pd.Series(r.values, S_o.index)

        return r

    def abs_return_backward(self, h:int)->pd.Series:
        """
        Backward return.
        Calculates the absolute price return over a period h. 

        Args:
            h: the horizon, in days, over which the return is calculated.

        Returns: 
            r: A series of absolute price returns.
        """
        idx=self.S.index.searchsorted(self.S.index-pd.Timedelta(days=h)) # Get the index positions.
        idx=idx[idx>0] # The first value in the price series does not have a backward difference.
        S_o=pd.Series(self.S.index[idx-1], index=self.S.index[self.S.shape[0]- idx.shape[0]:]) # Whats left is a series with less values than originally in the price series. 
                                                                                               # Since we are looking at the backward difference we will be missing returns corresponding to the beginning of the price series.
        r=self.S.loc[S_o.index]/self.S.loc[S_o.values].values-1 # Backward return.
        
        return r


    def dynamic_threshold(self, W:int, h:int, forward:bool)->pd.Series:
        """
        Calculates a threshold at each time bar. Based on an exponentially weighted moving standard deviation over a window size W.

        Args:
            W: The window size in days.  
            h: The horizon, in days, over which the return is calculated. 
            forward: If True then the forward return is returned. A False returns the backward return.

        Returns:
            T: A series of thresholds.

        """
        if forward is True:
            r = self.abs_return_forward(h)
        else:
            r = self.abs_return_backward(h)

        T = r.ewm(span=W).std()
        T=T.dropna()

        return T

    ###############################################################################################################################################################################

    def assign_label(self, x:float, T:float)->int:
        """
        Assign a label of 0, 1 or -1 to a set of features.

        Args:
            x: The price return.
            T: A fixed or dynamic threshold.

        Returns: 
            A label of 0, 1 or -1
        """
        if x < -T:
            return -1
        elif abs(x) <= T:
            return 0
        elif x > T:
            return 1


    def fixed_time_horizon(self,  h:int, dynamic:bool, T:float=None, W:int=None)->pd.DataFrame:
        """
        Fixed-time horizon labelling method. 

        Args: 
            h: The horizon, in days, over which returns are calculated.
            dynamic: A boolean variable. If True then dynamic thresholding will apply. If False then thespecified fixed threshold will apply. 
            T: A fixed threshold. Optional, depending on the dynamic variable.
            W: The window size, in days, used to calculate the dynamic threshold. Optional, depending on the dynamic variable.
            
        Returns: 
            df_out: A dataframe with columns 'retun', 'threshold', 'label'.

        """
        r = self.abs_return_forward(h)
        df_out = pd.DataFrame(r, columns=['return'], index=r.index)

        if dynamic is True:
            if W is None:
                raise ValueError(f'Please specify a value for parameter W.')
            T = self.dynamic_threshold(W,h,True)
            df_out['threshold'] = T 
            df_out['label'] = df_out.apply(lambda x: self.assign_label(x['return'], x['threshold']), axis=1)
        else: 
            if T is None:
                raise ValueError(f'Please specify a value for parameter T.')
            df_out['threshold'] = T
            df_out['label'] = df_out.apply(lambda x: self.assign_label(x['return'], T), axis=1)

        df_out=df_out.dropna()

        return df_out, df_out['threshold']

    ###############################################################################################################################################################################

    def vertical_barrier(self, events:list, expiry:int)->pd.Series:
        """
        Finds the vertical barriers corresponding to each event.

        Args:
            events: A list of datetime indices. 
            expiry: The number of days until expiry, i.e., the length of vertical barrier. Set to False if there are no vertical barriers.

        Returns: 
            v_barrier: A series of vertical barrier dates (data) with corresponding event dates (index).
        """
        idx_list = self.S.index.searchsorted(events+pd.Timedelta(days=expiry))  # Form a list of sorted indices that include the event and its vertical barrier. 
        idx_list = idx_list[idx_list<self.S.shape[0]] # Remove all indices that don't exist in the list of prices.
        v_barrier = pd.Series(self.S.index[idx_list], index=events[:idx_list.shape[0]]) # Series index - event dates. Series data - vertical barrier.
        if expiry is False: 
            v_barrier = pd.Series(pd.NaT, index=events) # Inactive vertical barrrier.
        return v_barrier

    def horizontal_barrier(self, T:pd.Series, events:list, pt_sl:list)->pd.Series:
        """
        Calculates the upper and lower horizontal barriers corresponding to each event.

        Args:
            T: A series of thresholds. 
            events: A list of datetime indices.
            pt_sl: A list of multiplication factors for the upper and lower barrier.

        Returns:
            pt: Profit taking (upper) barrier.
            sl: Stop loss (lower) barrier.
        """
        idx_list = T.index.isin(events)
        T = T.loc[idx_list]  # Get the thresholds at the dates corresponding to the sampled events.

        if pt_sl[0]>0: 
            pt=pt_sl[0]*T
        else:
            pt=pd.Series(index=events.index) # Inactive upper barrier. 
        if pt_sl[1]>0: 
            sl=-pt_sl[1]*T
        else:
            sl=pd.Series(index=events.index) # Inactive lower barrier.

        return pt,sl

    def barrier_touch(self, df_events:pd.DataFrame, pt:pd.Series, sl:pd.Series)->pd.DataFrame:
        """
        Finds the times at which each barrier was touched.

        Args: 
            df_events: A dataframe with columns 'v_barrier' and 'side'. 
            pt: A series of profit taking barriers.
            sl: A series of stop loss barriers.

        Returns:
            df_out: A dataframe with columns 'v_barrier', 'sl', 'pt'.
        """
        for loc,v_bar in df_events['v_barrier'].fillna(self.S.index[-1]).items():
            S_o=self.S[loc:v_bar] # Get the prices that span the width of the barrier.
            S_o=(S_o/self.S[loc]-1)*df_events.loc[loc,'side'] # Calculate the returns over this path.
            df_events.loc[loc,'sl']=S_o[S_o<sl[loc]].index.min() # Earliest stop loss. Lower barrier touched.
            df_events.loc[loc,'pt']=S_o[S_o>pt[loc]].index.min() # Earliest profit taking. Upper barrier touched.
            
        return df_events.drop('side', axis=1)


    def get_events(self,  T:pd.Series, v_barrier:pd.Series, events:list, pt_sl:list, min_ret:float, side:pd.Series)->pd.DataFrame:
        """
        Finds the earliest date that either one of the barriers were touched.

        Args:
            T: A series of thresholds corresponding to the horizontal barrier.
            v_barrier: A series dates at which the vertical barriers occur.
            events: A list of datetime indices.
            pt_sl: A list of multiplication factors applied to the horizontal barriers.
            min_ret: The minimum allowable price return. 
            side: A series with either 1s or -1s indicating the side of the trade. Set to False if the side is unknown.
               
        Returns: 
            df_out: A dataframe with columns 'date_touched' and 'barrier'.
        """
        if side is False: 
            side = pd.Series(1, index=T.index) # If the side is unknown set it to 1.
        pt,sl = self.horizontal_barrier(T, events, pt_sl) # Get the upper and lower horizontal barriers.
        df_out = pd.concat({'v_barrier':v_barrier, 'side':side}, axis=1).dropna() # Form a dataframe with the dates of the vertical barrier and the side for each event.
        df = self.barrier_touch(df_out, pt, sl) # Get the dates when each barrier was touched.
        df_out['date_touched'] = df.dropna(how='all').min(axis=1) # Find the earliest date.
        df_out['barrier'] = df.dropna(how='all').idxmin(axis=1)  # Find the barrier corresponding to the earliest date.
        df_out=df_out.drop(['side','v_barrier'], axis=1) 
        return df_out
   

    def triple_barrier(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Triple-barrier labelling method.

        Args:
            df: A dataframe with columns 'date touched' and 'barrier'. 
        
        Returns: 
            df_out: A dataframe with columns 'date touched', 'barrier', 'return', 'label'.
        """
        
        df_out = pd.DataFrame(df.date_touched.values, columns=['date touched'], index=df.index) 
        df_out['barrier']  = df.barrier
        df_out['return'] = self.S.loc[df.date_touched.values].values/self.S.loc[df.index]-1 # Forward return.
        df_out['label'] = df_out.apply(lambda x: 0 if x['barrier']=='v_barrier' else np.sign(x['return']), axis=1 ) # Assign 0 to a label if the vertical barrier was touched first.
                                                                                                                    # Otherwise use the sign of the price return as the label value. 

        return df_out
        

    #################################################################################################################################################################################

    def num_conc_events(self,date_touched:pd.Series)->pd.Series:
        """
        Counts the number of labels at each time bar.

        Args:
            date_touched: A series of times when the barrier was touched.

        Returns: 
            count: A series of the number of labels/events occuring at each time
        """
        count = pd.Series(0, self.S.index)

        # Increment count over the time period that the labels exist over.
        for t0,t1 in date_touched.items():
            count.loc[t0:t1] = count.loc[t0:t1] +1

        return count

    def uniqueness(self, num_conc:pd.Series, date_touched:pd.Series)->pd.Series:
        """
        Calculates the average uniqueness of a label/event.

        Args:
            num_conc: A series containing the number of labels at each time bar.
            date_touched: A series of dates when the barrier was touched.

        Returns: 
            u_ave: The average uniqueness of each label/event.

        """
        u_ave = pd.Series(0, index=date_touched.index)

        # A label's average uniqueness is the sum of the label's uniqueness over all time divided by the number of time bars it exists over.     
        # Equivalent to the reciprocal of the harmonic mean of the number of concurrent events over the time it exists.
        for t0,t1 in date_touched.items():
            u_ave.loc[t0] = (1/num_conc[t0:t1]).sum()/(num_conc[t0:t1].shape[0])
            
        return u_ave 

    def weights_returns(self, num_conc:pd.Series, date_touched)->pd.Series:
        """
        Calculates the label weighting factors according to unique price returns.

       Args:
            num_conc: A series containing the number of labels at each discrete time.
            date_touched: A series of times when the barrier was touched.

        Returns:
            weights: A series of weighting factors.

        """
        weights_abs = pd.Series(0,index=date_touched.index)
    
        # Absolute weight of each label, calculated by summing scaled returns (according to uniquness) at each time bar where the label exists.
        for t0,t1 in date_touched.items():
            weights_abs.loc[t0] = np.abs(((self.S.loc[t0:t1]/self.S.loc[t0] + 1)/num_conc[t0:t1]).sum())
        
        # Scale the weights to add up to the total number of labels/events.
        weights = weights_abs*weights_abs.shape[0]/(weights_abs.sum())

        return weights

    def weights_decay(self, uniqueness:pd.Series, weight_old:float)->pd.Series:
        """
        Calcuates the decay weighting factors. Weights are assigned linearly starting from the latest to the oldest oberservation
        according to the linear function: weights(u) = b*u + a.

        Args:
            uniqueness: The average uniqueness of a label/event.
            weight_old: The weight of the oldest observation in time.

        Returns: 
            weights: A series of weighting factors associated with the cumulative uniqueness.

        Raises:
            ValueError: If the oldest weight does not exist in the interval (-1,1].

        """
        if not -1<weight_old<=1:
            raise ValueError(f"The oldest weight must exist in the interval (-1,1].")

        u_cumsum = uniqueness.cumsum()  # Cumulative sum of the uniqueness.
        
        # Find the a and b parameters of the straight line function.
        if weight_old >=0:
            b = (1-weight_old)/u_cumsum.iloc[-1] 
        else:
            b = 1/((1+weight_old)*u_cumsum.iloc[-1])
        a =  1 - b*u_cumsum.iloc[-1]

        weights = a + b*u_cumsum # Interpolation.
        weights[weights<0]=0 # Set negative weights to zero. These observations do not contribute.

        return weights