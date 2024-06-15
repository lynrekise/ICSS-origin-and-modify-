## Author: Lynrekise
## Time: 2024-6-15
## content: Iterated Cumulative Sums of Squares (ICSS) Algorithm and Modify ICSS Algorithm
## python version: 3.7.9
## numpy version: 1

import numpy as np
import pandas as pd

class ICSS:
    def __init__(self, data, demean=False):
        # data: np.array or array-like
        if demean: # centerite the data
            data = data - np.mean(data)
        if not isinstance(data,np.ndarray):
            self.data = np.array(data)

    def icss(self):
        data = self.data
        # initialize the detection series
        # S is a stack recording the index of sequence(start from 1)
        S = [[1,len(data)]]

        # initialize the potiential change points
        potiential_change_points = np.zeros(len(data))
        cp_index = 0

        while(len(S)!=0):
            current_range = S.pop()
            change_points_sub = self.icss_step_1_and_2(data[current_range[0]-1:current_range[1]])

            if len(change_points_sub)==2:
                S.append(list(change_points_sub + current_range[0]))

            if len(change_points_sub)>0:
                potiential_change_points[cp_index:(cp_index+len(change_points_sub))] = change_points_sub + current_range[0]

            cp_index += len(change_points_sub)

        if np.nansum(potiential_change_points)==0:
            return "Unable to identify structural variance breaks in the series."

        potiential_change_points = np.sort(np.unique(np.append(potiential_change_points,[0,len(data)])))

        converged = False
        new_cps_stack = []


        print('==============')
        while(not converged):
            for i in range(1,len(potiential_change_points)-1):
                start = int(potiential_change_points[i-1] + 1)
                end = int(potiential_change_points[i+1])

                if np.size(end)==0:
                    return np.nan

                D_k = self.centered_cusum_values(data[start:end])
                tmp = self.check_critical_value(D_k)

                if np.isnan(tmp['position']):
                    return np.nan
                else:
                    exceeds = tmp['exceeds']
                    position = tmp['position']

                if exceeds:
                    new_cps_stack.append(start+position)

            stack_entries = new_cps_stack[::-1]

            new_cps = np.sort(np.unique(np.append(stack_entries,[0,len(data)])))
            converged = self.is_converged(potiential_change_points,new_cps)

            if not converged:
                potiential_change_points = new_cps

        change_points = potiential_change_points[1:len(potiential_change_points)-1]
        return change_points


    def icss_step_1_and_2(self, time_series):
        change_points = []

        if np.size(time_series)==0:
            return np.nan

        D_k = self.centered_cusum_values(time_series)
        tmp = self.check_critical_value(D_k)
        if np.isnan(tmp['position']):
            return np.nan
        exceeds = tmp['exceeds']
        position_step1 = tmp['position']

        if exceeds:
            position = position_step1

            while exceeds:
                global t2
                t2 = position
                D_k_step2a = self.centered_cusum_values(time_series[:t2])
                tmp = self.check_critical_value(D_k_step2a)
                if np.isnan(tmp['position']):
                    return np.nan
                else:
                    exceeds = tmp['exceeds']
                    position = tmp['position']

            k_first = t2

            position = position_step1 + 1
            exceeds = True

            while exceeds:
                global t1
                t1 = position
                D_k_step2b = self.centered_cusum_values(time_series[t1-1:])
                tmp = self.check_critical_value(D_k_step2b)
                if np.isnan(tmp['position']):
                    return np.nan
                else:
                    exceeds = tmp['exceeds']
                    position2 = tmp['position']
                    position = position2 + position

            k_last = t1 - 1

            if k_first==k_last:
                change_points = np.array([k_first])
            else:
                change_points = np.array([k_first,k_last])

        return change_points

    def centered_cusum_values(self, time_series):
        T = len(time_series)
        squared = np.square(time_series)
        C_k = np.cumsum(squared)
        C_T = C_k[T-1]

        D_k = C_k/C_T - np.arange(1,T+1)/T

        return D_k

    def check_critical_value(self, D_k):
        # position is a time series index
        value = max(abs(D_k))
        position = np.where(abs(D_k)==value)[0]

        if len(position)>1:
            return {'positon': np.nan, 'exceeds': np.nan}

        M = np.sqrt(len(D_k)/2) * value
        exceeds = (M>1.358)

        return {'position': position[0]+1, 'exceeds': exceeds}

    def is_converged(self, old, new):
        if len(old)==len(new):
            for i in range(len(new)):
                low = min(old[i],new[i])
                high = max((old[i],new[i]))
                if (high-low)>2:
                    return False
        else:
            return False

        return True


class ModifyICSS:
    def __init__(self, data, demean=False):
        # data: np.array or array-like
        if demean: # centerite the data
            data = data - np.mean(data)
        if not isinstance(data,np.ndarray):
            self.data = np.array(data)

    def icss(self):
        data = self.data
        # initialize the detection series
        # S is a stack recording the index of sequence(start from 1)
        S = [[1,len(data)]]

        # initialize the potiential change points
        potiential_change_points = np.zeros(len(data))
        cp_index = 0

        while(len(S)!=0):
            current_range = S.pop()
            change_points_sub = self.icss_step_1_and_2(data[current_range[0]-1:current_range[1]])

            if len(change_points_sub)==2:
                S.append(list(change_points_sub + current_range[0]))

            if len(change_points_sub)>0:
                potiential_change_points[cp_index:(cp_index+len(change_points_sub))] = change_points_sub + current_range[0]

            cp_index += len(change_points_sub)

        if np.nansum(potiential_change_points)==0:
            return "Unable to identify structural variance breaks in the series."

        potiential_change_points = np.sort(np.unique(np.append(potiential_change_points,[0,len(data)])))

        converged = False
        new_cps_stack = []


        print('==============')
        while(not converged):
            for i in range(1,len(potiential_change_points)-1):
                start = int(potiential_change_points[i-1] + 1)
                end = int(potiential_change_points[i+1])

                if np.size(end)==0:
                    return np.nan

                D_k = self.GK(data[start:end])
                tmp = self.check_critical_value(D_k)

                if np.isnan(tmp['position']):
                    return np.nan
                else:
                    exceeds = tmp['exceeds']
                    position = tmp['position']

                if exceeds:
                    new_cps_stack.append(start+position)

            stack_entries = new_cps_stack[::-1]

            new_cps = np.sort(np.unique(np.append(stack_entries,[0,len(data)])))
            converged = self.is_converged(potiential_change_points,new_cps)

            if not converged:
                potiential_change_points = new_cps

        change_points = potiential_change_points[1:len(potiential_change_points)-1]
        return change_points


    def icss_step_1_and_2(self, time_series):
        change_points = []

        if np.size(time_series)==0:
            return np.nan

        D_k = self.GK(time_series)
        tmp = self.check_critical_value(D_k)
        if np.isnan(tmp['position']):
            return np.nan
        exceeds = tmp['exceeds']
        position_step1 = tmp['position']

        if exceeds:
            position = position_step1

            while exceeds:
                global t2
                t2 = position
                D_k_step2a = self.GK(time_series[:t2])
                tmp = self.check_critical_value(D_k_step2a)
                if np.isnan(tmp['position']):
                    return np.nan
                else:
                    exceeds = tmp['exceeds']
                    position = tmp['position']

            k_first = t2

            position = position_step1 + 1
            exceeds = True

            while exceeds:
                global t1
                t1 = position
                D_k_step2b = self.GK(time_series[t1-1:])
                tmp = self.check_critical_value(D_k_step2b)
                if np.isnan(tmp['position']):
                    return np.nan
                else:
                    exceeds = tmp['exceeds']
                    position2 = tmp['position']
                    position = position2 + position

            k_last = t1 - 1

            if k_first==k_last:
                change_points = np.array([k_first])
            else:
                change_points = np.array([k_first,k_last])

        return change_points

    def GK(self, time_series):
        T = len(time_series)
        squared = np.square(time_series)
        C_k = np.cumsum(squared)
        C_T = C_k[T-1]

        GK_right = C_k - (np.arange(1,T+1)/T)*C_T
        m = int(4 * ((T/100)**(1/4)))+1
        sigma_hat_square = np.var(time_series)
        omega_left = (1/T)*np.sum(np.square(squared-sigma_hat_square))
        omega_right = 0
        for l in range(1,m+1):
            w_l_m = 1 - l/(m+1)
            diff_var = 0
            for t in range(l,T):
                diff_var += (time_series[t]**2 - sigma_hat_square)*(time_series[t-1]**2 - sigma_hat_square)
            omega_right += w_l_m*diff_var
        omega_right = (2/T)*omega_right
        omega_hat = omega_right + omega_left
        GK_left = 1/np.sqrt(omega_hat)
        G_k = GK_left * GK_right

        return G_k

    def check_critical_value(self, G_k):
        T = len(G_k)
        # position is a time series index
        value = max(abs(G_k))
        position = np.where(abs(G_k)==value)[0]

        if len(position)>1:
            return {'positon': np.nan, 'exceeds': np.nan}

        M = value / np.sqrt(len(G_k))
        critical_value = 1.405828+(-3.317278)*(T**(-0.5))+31.22133*(T**(-1))+(-1672.206)*(T**(-2))+(52870.53)*(T**(-3))+(-411015)*(T**(-4))
        exceeds = (M>critical_value)

        return {'position': position[0]+1, 'exceeds': exceeds}

    def is_converged(self, old, new):
        if len(old)==len(new):
            for i in range(len(new)):
                low = min(old[i],new[i])
                high = max((old[i],new[i]))
                if (high-low)>2:
                    return False
        else:
            return False

        return True