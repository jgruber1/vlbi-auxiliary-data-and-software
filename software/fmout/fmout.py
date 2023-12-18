#!/usr/bin/env python3
#
# Copyright (c) 2021, 2022 WHI LLC
#
# fmout: Extract fmout and maser data from FS logs
# (see http://github.com/whi-llc/fmout).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import os
import sys
import getopt
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import warnings
import string

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
def breaks(x,y,step,list):
    times= [x[0]-1]
    last=y[0]
    if step > 0:
        for i in range(1,x.size):
            if math.fabs(y[i]-last) > step:
                times.append(0.5*(x[i]+x[i-1]))
#above makes the jump epoch midway between the latest two epochs
#the next (comment) line is for one second before the last value, but
#                times.append(np.maximum(0.5*(x[i]+x[i-1]),x[i]-1))
#really need it before the latest scan
            last=y[i]
    times.append(x[x.size-1]+1)
    times=times+list
    times=sorted(times)
    return times

fit_line  = True
make_plot = False
all_data = False
epoch_begin = False
epoch_fixed = 0
plot_files = False
print_cov = False
save_raw = False
use_line = False
merge_data = False
trim_points = False
delete_points = False
wrap_points = False
use_limit = True
debug_output = True
regex_selection = False
regex_invert = False
find_jumps = False
plot_raw_data = False
delta_jumps = False
remove_points = False
orig_epoch = False

if len(sys.argv)==1:
    sys.exit('try '+sys.argv[0]+' -h')

try:
    options, remainder = getopt.getopt(
    sys.argv[1:],
    'abcd:e:fg:hj:lm:nopqr:s:t:uvwyzDR')

except getopt.GetoptError as err:
    print(('ERROR:', err))
    sys.exit(1)

for opt,arg in options:
    if opt == '-h':
        print('Usage: '+os.path.basename(os.path.splitext(sys.argv[0])[0])+' -abcd:ef:g:hj:lm:nopqr:s:t:uvwyzDR files')
        print('''
 fits fmout & maser data from log files, prints values in order:
    log name
    data type
    offset (microseconds)
    rate (sec/sec)
    epoch of offset to the nearest second
    sigma of samples (microseconds)
 the sign of offset and rate is reversed if the data type ends
 in fmout or maser, i.e., values are "clock_early"
Options:
 -a   plot all data, otherwise start after first scan_name= and preob
      and end before the last postob, per file
 -b   fit for offset at beginning of data,
      otherwise at the centroid or overridden by "-e"
 -c   print covariance, otherwise omitted, values in order:
        offset sigma (scaled)
        rate   sigma (scaled)
        correlation
        included points / total points
        deviation of residuals: (max-min)/sigma
      sigmas and correlation are not available for "-j" and
        some versions of numpy
      residual plots show the model variance
 -d value (possibly float)
      delete points more than "value" ms from zero, "500" might be useful
      this is applied after "-w" and before "-t"
 -e epoch
      epoch format: 2017y198d22h12m08s
      fields may be omitted from the right through hours
      this option overrides "-b"
 -f   generate plot files ending .pdf
 -g string
      Python regex expression to select what data is used
      e.g., "fmout" would select only fmout data
 -h   this text
 -j  values (possibly floats) separated by colons (no white space)
      the first value is the size of a jump (microseconds) to be considered an
        offset break, "0.4" might be useful, a negative value disables jump
        detection
      the remaining values either explicit epochs and/or values in seconds
      relative to the overall epoch for additional offsets:
        for explicit epochs, the value is in the format of "-e" with all
        fields specified, additionally a fixed epoch must have been specified
        with "-e" previously in the command line
        for values in seconds, the value is relative to the overall epoch
          (see "-b" and "-e"), the times can be read interactively off the
          "-p" plot with the pointer and multiplied by the appropriate scale,
          3600 for hours or 86400 for days
      this option implies "-b", unless "-e" is specified
      the offset value for the break in the text output is at the epoch of the
      break unless "-o" or "-q" is specified
      in the plot output ("-f" and "-p"), the epochs are indicated by vertical
      lines, the offset reported in plot annotations is the initial offset
 -l   connect data with a line in plots
 -m string
      merge all the input files using "string" to identify the result
 -o   print the offset values for breaks (see "-j") at the original epoch,
      instead of at the break epochs
 -n   no line fitting
      implies "-u"
 -p   display plots interactively
 -q   print the deltas (changes) in the offset for breaks (see "-j")
 -r value
      remove residuals more than "value" ns from median, "50" might be useful
      this is applied after "-w" and "-d" and "-t"
      it is best to remove the biggest outliers with other flags, the add "-r"
 -s value
      generate simulated data with RMS noise "value" in ns, written to standard
      output, which can be redirected to a file and then processed with fmout
        if "-s 0" is used to generate the data,
        the output processed with "-abc" should produce:
                -2.500 -3.500e-12 2017y278d18h00m00s 0.000
              8.83e-17  1.532e-27   -0.866 500 / 500   3.1
        if "-s 10" is used to generate the data,
        the output processed with "-abc" should repeatably produce:
                -2.500 -3.493e-12 2017y278d18h00m00s 0.010
              8.79e-04  1.524e-14   -0.866 500 / 500   7.2
 -t value
      trim points more than "value" ms from median, "1" might be useful
      "value" can be a float
      this is applied after "-w" and "-d"
 -u   plot original data and any model
 -v   print version and exit
 -w   wrap values more than 0.5 seconds from zero
      this option is applied before "-d" and "-t"
 -y   accept points outside +- 1 second
      if not, points outside +- 1 second are removed before considering "-wdt"
 -z   invert the sense of the "-g" selection
 -D   suppress debug error messages: no data, all data deleted, Unicode, etc.
 -R   save pre-fit data to files ending .dat
      each line contains time-in-seconds-from-epoch followed by value''')
        sys.exit(0)
    elif opt == '-a':
        all_data = True
    elif opt == "-b":
        epoch_begin = True
    elif opt == "-c":
        print_cov = True
    elif opt == "-t":
        trim_points = True
        trim_value = float(arg)*1000
    elif opt == "-d":
        delete_points = True
        delete_value = float(arg)*1000
    elif opt == "-r":
        remove_points = True
        remove_value = float(arg)*0.001
    elif opt == '-z':
        regex_invert = True
    elif opt == '-g':
        regex_selection = True
        regex_object = re.compile(arg)
    elif opt == "-j":
        find_jumps = True
        arg_list=arg.split(':')
        jump_value=float(arg_list[0])
        jump_list = []
        for i in range(1,len(arg_list)):
            try:
                epoch_jump=datetime.datetime.strptime(arg_list[i],'%Yy%jd%Hh%Mm%Ss')
            except ValueError:
                jump_list.append(float(arg_list[i]))
            else:
                if epoch_fixed == 0:
                     sys.exit('if using an eppch for "-j", must use "-e" before "-j" in argument list')
                else:
                    jump_list.append((epoch_jump-epoch_fixed).total_seconds())
        epoch_begin = True
    elif opt == '-R':
        save_raw = True
    elif opt == '-l':
        use_line = True
    elif opt == '-m':
        merge_data = True
        merge_string = arg
    elif opt == '-n':
        fit_line = False
        plot_raw_data = True
    elif opt == '-D':
        debug_output = False
    elif opt == '-o':
        if delta_jumps:
            sys.exit(" -o and -q can't be used together")
        orig_epoch = True
    elif opt == '-p':
        make_plot = True
    elif opt == '-q':
        if orig_epoch:
            sys.exit(" -o and -q can't be used together")
        delta_jumps = True
    elif opt == '-f':
        plot_files = True
    elif opt == '-u':
        plot_raw_data = True
    elif opt == '-v':
        sys.exit('[Version 0.88]')
    elif opt == '-w':
        wrap_points = True
    elif opt == '-y':
        use_limit= False
    elif opt == "-s":
        value = float(arg)*1e-9
        if value < 1e-16:
            value=1e-16
        d = datetime.datetime(2017, 10, 5, 18, 00)
        off = -2.5e-6
        rate = -3.5e-12
        np.random.seed(42) # the answer to the question
        for n in range(500):
            b = d + n*datetime.timedelta(0,200)
            print(b.strftime('%Y.%j.%H:%M:%S.%f')[:-4]+'/fmout-gps/'+'{:e}'.format(off+n*200*rate+np.random.normal(loc=0.0,scale=value)))
        sys.exit(0)
    elif opt == '-e':
        try:
            epoch_fixed = datetime.datetime.strptime(arg,'%Yy%jd%Hh%Mm%Ss')
        except ValueError:
            try:
                epoch_fixed = datetime.datetime.strptime(arg,'%Yy%jd%Hh%Mm')
            except ValueError:
                try:
                    epoch_fixed = datetime.datetime.strptime(arg,'%Yy%jd%Hh')
                except ValueError:
                    try:
                        epoch_fixed = datetime.datetime.strptime(arg,'%Yy%jd')
                    except ValueError:
                        sys.exit('bad epoch format for -e')

iterarg = iter(remainder)

fmout = re.compile(r'^([0-9.:]{20})(?:|;"|[^;][^"].*)/(.*fmout.*)/[^-+\.\d]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\S*)')
maser = re.compile(r'^([0-9.:]{20})(?:|;"|[^;][^"].*)/(.*maser.*)/[^-+\.\d]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\S*)')
rdbe_gps = re.compile(r'^([0-9.:]{20})#(rdtc.#dot2.ps.*)/(\S+)')
#2017.278.18:00:00.16#rdtcc#dot2gps/1.157031250e-05
#12345678901234567890
#2017.337.22:47:46.86/rdbeb/!dbe_gps_offset?0:-5.189453125e-05;
rdbe_gps2 = re.compile(r'^([0-9.:]{20})/(rdbe.)/!(dbe_.ps_offset)\?0:([^;]+);')
dbbcout = re.compile(r'^([0-9.:]{20})(?:|;"|[^;][^"].*)/(.*dbbcout.*)/[^-+\.\d]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\S*)')
#
scan_name= re.compile(r'^([0-9.:]{20}):scan_name=');
preob= re.compile(':preob')
postob= re.compile(':postob')
surrogates = re.compile(r"[\uDC80-\uDCFF]")

for arg in iterarg:
    if all_data:
        scan_name_found=1
        preob_found=1
    else:
        scan_name_found=0
        preob_found=0
    if not merge_data or arg == remainder[0]:
        x={}
        y={}
        count={}
        last = {}
        negative={}
        first_time = 0
    lines=0
    with open(arg, errors="surrogateescape") as infile:
        for line in infile:
            lines=lines+1
            if surrogates.search(line):
                if debug_output:
# new lines may not be interpreted correctly, hence approximately
                    eprint(f"Found Unicode error on line {lines} (approximately)")
                continue
            if first_time == 0:
                try:
                    first_time=datetime.datetime.strptime(
                        line[0:20],'%Y.%j.%H:%M:%S.%f')
                except ValueError:
                    if debug_output:
                        eprint('Bad first time in "'
                            +arg+'": "'+line.strip()+'"')
                        continue
            if not scan_name_found:
                m=scan_name.search(line)
                if m:
                    scan_name_found=1
                    continue
            if  not preob_found:
                m=preob.search(line)
                if m:
                    preob_found=1
                    continue
            if not preob_found or  not scan_name_found:
                continue
            m=postob.search(line)
            if m:
                key_list=list(x.keys())
                for key in key_list:
                    last[key]=count[key]
            m=fmout.match(line)
            m2=maser.match(line)
            m3=rdbe_gps.match(line)
            m4=rdbe_gps2.match(line)
            m5=dbbcout.match(line)
            if m:
                key=m.group(2)
                if regex_selection:
                    g=regex_object.search(key)
                    if regex_invert:
                        if g:
                            continue
                    elif not g:
                        continue
                dt=datetime.datetime.strptime(m.group(1),'%Y.%j.%H:%M:%S.%f')
                try:
                    fm=float(m.group(3))*1e6
                except ValueError:
                    if debug_output:
                        eprint("can't decode line "+line.strip())
                    continue
                if key not in x:
                    count[key]=0
                    x[key]={}
                    y[key]={}
                    g = re.compile(r'gps[-2]fmout')
                    if g.search(key):
                        negative[key]=True
            elif m2:
                key=m2.group(2)
                if regex_selection:
                    g=regex_object.search(key)
                    if regex_invert:
                        if g:
                            continue
                    elif not g:
                        continue
                dt=datetime.datetime.strptime(m2.group(1),'%Y.%j.%H:%M:%S.%f')
                factor=1e6
# for shanghai
                if m2.group(4) == 'u':
                    factor=1
                try:
                    fm=float(m2.group(3))*factor
                except ValueError:
                    if debug_output:
                        eprint("can't decode line "+line.strip())
                    continue
                if key not in x:
                    count[key]=0
                    x[key]={}
                    y[key]={}
                    g = re.compile(r'gps[-2]maser')
                    if g.search(key):
                        negative[key]=True
            elif m3:
                key=m3.group(2)
                if regex_selection:
                    g=regex_object.search(key)
                    if regex_invert:
                        if g:
                            continue
                    elif not g:
                        continue
                dt=datetime.datetime.strptime(m3.group(1),'%Y.%j.%H:%M:%S.%f')
                try:
                    fm=float(m3.group(3))*1e6
                except ValueError:
                    if debug_output:
                        eprint("can't decode line "+line.strip())
                    continue
                fm=float(m3.group(3))*1e6
                if key not in x:
                    count[key]=0
                    x[key]={}
                    y[key]={}
            elif m4:
                key=m4.group(2) + '_' + m4.group(3)
                if regex_selection:
                    g=regex_object.search(key)
                    if regex_invert:
                        if g:
                            continue
                    elif not g:
                        continue
                dt=datetime.datetime.strptime(m4.group(1),'%Y.%j.%H:%M:%S.%f')
                try:
                    fm=float(m4.group(4))*1e6
                except ValueError:
                    if debug_output:
                        eprint("can't decode line "+line.strip())
                    continue
                fm=float(m4.group(4))*1e6
                if key not in x:
                    count[key]=0
                    x[key]={}
                    y[key]={}
            elif m5:
                key=m5.group(2)
                if regex_selection:
                    g=regex_object.search(key)
                    if regex_invert:
                        if g:
                            continue
                    elif not g:
                        continue
                dt=datetime.datetime.strptime(m5.group(1),'%Y.%j.%H:%M:%S.%f')
                try:
                    fm=float(m5.group(3))*1e6
                except ValueError:
                    if debug_output:
                        eprint("can't decode line "+line.strip())
                    continue
                if key not in x:
                    count[key]=0
                    x[key]={}
                    y[key]={}
                    g = re.compile(r'gps[-2]dbbcout')
                    if g.search(key):
                        negative[key]=True
            else:
                continue
            if use_limit and math.fabs(fm) >= 1e6:
                continue
            if wrap_points:
                if fm > 5e5:
                    fm=fm-1e6
                elif fm < -5e5:
                    fm=fm+1e6
            if delete_points:
                if abs(fm) >= delete_value:
                    continue
            if key in negative:
                fm=-fm
            x[key][count[key]]=dt
            y[key][count[key]]=fm
            count[key]=count[key]+1
#
# fit the data
#
    if merge_data and arg != remainder[-1]:
        continue
    if  not all_data:
        key_list=list(x.keys())
        for key in key_list:
            count[key]=last[key]
    all = {}
    key_list=list(x.keys())
    if debug_output:
        if not key_list:
            eprint('No data found in '+arg)
    for key in key_list:
        if count[key]==0:
            if debug_output:
                eprint('No data included for key '+key+' in '+arg)
            continue
#edit
        all[key]=count[key]
        if trim_points:
            o=np.array([])
            for i in range(count[key]):
                o=np.append(o,y[key][i])
            median=np.median(o)
            out=0
            for i in range(count[key]):
                if math.fabs(o[i]-median) < trim_value:
                    x[key][out]=x[key][i]
                    y[key][out]=y[key][i]
                    out=out+1
            count[key]=out
#remove points
        if remove_points and count[key]>1:
            t=np.array([])
            o=np.array([])
            for i in range(count[key]):
                t=np.append(t,(x[key][i]-x[key][0]).total_seconds())
                o=np.append(o,y[key][i])
            if count[key]==2:
                warnings.simplefilter('ignore', np.RankWarning)
                fit, _, _, _, _ = np.polyfit(t,o,1,full=True)
            else:
                warnings.simplefilter('always', np.RankWarning)
                fit, res, _, _, _ = np.polyfit(t,o,1,full=True)
            m,b=fit
            v = o - np.polyval(fit, t)
            out=0
            for i in range(count[key]):
                if math.fabs(v[i]) < remove_value:
                    x[key][out]=x[key][i]
                    y[key][out]=y[key][i]
                    out=out+1
            count[key]=out

#set epoch
        if count[key]==0:
            if debug_output:
                eprint('All data trimmed/removed for key '+key+' in '+arg)
            continue
        if epoch_fixed !=0:
            epoch=epoch_fixed
        elif epoch_begin:
            t=np.array([])
            for i in range(count[key]):
                t=np.append(t,(x[key][i]-x[key][0]).total_seconds())
            index=np.argsort(t)
            epoch=x[key][index[0]]
        else:
            average=0
            for i in range(count[key]):
                average=average+(x[key][i]-x[key][0]).total_seconds()
            epoch=x[key][0]+datetime.timedelta(0,average/count[key])
#
        t=np.array([])
        o=np.array([])
        for i in range(count[key]):
            t=np.append(t,(x[key][i]-epoch).total_seconds())
            o=np.append(o,y[key][i])
#sort
        index=np.argsort(t)
        t=t[index]
        o=o[index]
# plot time axis
        diff=t[count[key]-1]-t[0]
        days=diff>86400*2
        u=np.array([])
        for i in range(count[key]):
            if days:
                u=np.append(u,t[i]/86400)
            else:
                u=np.append(u,t[i]/3600)
# find breaks
        cov_available = False
        if find_jumps:
            times=breaks(t,o,jump_value,jump_list)
            l_times=len(times)
            ar=np.zeros((t.size,l_times))
            for i in range(t.size):
                ar[i][l_times-1]=t[i]
                for j in range(l_times-1):
                    if t[i] >times[j] and t[i] <times[j+1]:
#                    if t[i] >times[j]:
                        ar[i][j]=1
            coeff,res,rank,_ = np.linalg.lstsq(ar,o,rcond=None)
            if rank < l_times:
                sys.exit("breaks made fit degenerate, are they relative to the wrong epoch or too close together?")
            res=math.sqrt(res/(count[key]-l_times))
            v = o - np.dot(ar,coeff)
# fit init
        elif fit_line:
            v=[0]
            b=o[0]
            m=0
            res=0
#fit
            if count[key]>1:
                if count[key]==2:
                    warnings.simplefilter('ignore', np.RankWarning)
                    fit, _, _, _, _ = np.polyfit(t,o,1,full=True)
                else:
                    warnings.simplefilter('always', np.RankWarning)
                    fit, res, _, _, _ = np.polyfit(t,o,1,full=True)
                m,b=fit
                if remove_points:
                    v = o - np.polyval(fit, t)
#
                if count[key]>2 and print_cov:
                    try:
                        _, cov = np.polyfit(t,o,1, cov=True)
                        cov_available = True
                        offset_cov=math.sqrt(max(0,cov[1][1]))
                        rate_cov=math.sqrt(max(0,cov[0][0]))
                        if offset_cov !=0 and rate_cov!=0:
                            corr_cov= cov[0][1]/(offset_cov*rate_cov)
                    except TypeError:
                        pass
#
                v = o - np.polyval(fit, t)
                if count[key] > 2:
                    res=math.sqrt(res/(count[key]-2))
# residuals
        if find_jumps or fit_line:
            mx=v[0]
            mn=v[0]
            for i in range(count[key]):
                if v[i] > mx:
                    mx=v[i]
                if v[i] < mn:
                    mn=v[i]
#
            if res > 0:
                dev=(mx-mn)/res
            else:
                dev=0
#
        tim=epoch.strftime('%Yy%jd%Hh%Mm%Ss')
#        tim=epoch.strftime('%Yy%jd%Hh%Mm%S.%f')
# remove trail zeros in fractional seconds
#        for i in range(4):
#            if tim[-1] == '0':
#                tim=tim[:-1]
#            else:
#                break
#        tim=tim+'s'
# find name
        if not merge_data:
            head, tail =os.path.split(arg)
            root, ext =os.path.splitext(tail)
        else:
            root = merge_string
        if save_raw:
            dataf=open(root+'_'+key+'.dat','w')
            for i in range(count[key]):
                dataf.write(str(t[i])+' '+str(o[i])+'\n')
            dataf.close()
#print data
        if find_jumps:
            print('{:9s}'.format(root), '{:24s}'.format(key), end=' ')
            m=coeff[-1]
            b=coeff[0]
            print('{:8.3f}'.format(b), '{:10.3e}'.format(m*1e-6),tim, end=' ')
            if res < 0.001:
                print('{:9.3e}'.format(res))
            else:
                print('{:.3f}'.format(res))
            for i in range(1,l_times-1):
                print('{:9s}'.format(root), end=' ')
                if delta_jumps:
                    print('{:24s}'.format('break'), end=' ')
                elif orig_epoch:
                    print('{:24s}'.format('break (original epoch)'), end=' ')
                else:
                    print('{:24s}'.format('break (break epoch)'), end=' ')
                tim=(epoch+datetime.timedelta(
                    0,times[i])).strftime('%Yy%jd%Hh%Mm%Ss')
                if delta_jumps:
                    b1=coeff[i]-coeff[i-1]
                    print('{:8.3f}'.format(b1), '{:10s}'.format('delta'),tim)
                else:
                    if orig_epoch:
                        b1=coeff[i]
                    else:
                        b1=coeff[i]+times[i]*m
                    print('{:8.3f}'.format(b1), '{:10s}'.format('total'),tim)
        elif fit_line:
            print('{:9s}'.format(root), '{:24s}'.format(key), end=' ')
            print('{:8.3f}'.format(b), '{:10.3e}'.format(m*1e-6),tim, end=' ')
            if res < 0.001:
                print('{:9.3e}'.format(res))
            else:
                print('{:.3f}'.format(res))
        if find_jumps or fit_line:
            if print_cov:
                print('{:9s}'.format(root), '{:24s}'.format(key), end=' ')
                if cov_available:
                    print('{:7.2e}'.format(offset_cov), end=' ')
                    print('{:10.3e}'.format(rate_cov*1e-6), end=' ')
                    print('{:8.3f}'.format(corr_cov), end=' ')
                else:
                    print('    -  ', end=' ')
                    print('   -      ', end=' ')
                    print('    -   ', end=' ')
                print(count[key], end=' ')
                print('/',all[key], end=' ')
                print('{:5.1f}'.format(dev))
# plotting
        if make_plot or plot_files:
# plot time axis
            diff=t[count[key]-1]-t[0]
            days=diff>86400*2
            u=np.array([])
            for i in range(count[key]):
                if days:
                    u=np.append(u,t[i]/86400)
                else:
                    u=np.append(u,t[i]/3600)
            u_times=np.array([])
            if find_jumps:
                for i in range(l_times):
                    if days:
                        u_times=np.append(u_times,times[i]/86400)
                    else:
                        u_times=np.append(u_times,times[i]/3600)
#
            plot = plt.figure(figsize=(8,6))
            if plot_raw_data:
                plt.plot(u,o,'b.')
                if use_line:
                    plt.plot(u,o,'b-')
                if find_jumps:
                    plt.plot(u,np.dot(ar,coeff),'k-')
                elif fit_line:
                    plt.plot([u[0],u[-1]], m*np.array([t[0],t[-1]]) + b, 'k-')
            else:
                plt.plot(u,v,'b.')
                if use_line:
                    plt.plot(u,v,'b-')
#
                if cov_available:
                    msig=np.array([])
                    msign=np.array([])
                    u2=np.array([])
                    xmin,xmax,_,_=plt.axis()
                    xmin=t[0]
                    xmax=t[count[key]-1]
                    xdiff=(xmax-xmin)/499
                    for i in range(500):
                        part0=xmin+i*xdiff
                        u2=np.append(u2,part0/3600)
                        part1=1
                        prop=part0*part0*cov[0][0]
                        prop=prop+part0*part1*cov[0][1]
                        prop=prop+part1*part0*cov[1][0]
                        prop=prop+part1*part1*cov[1][1]
                        prop=math.sqrt(prop)
                        msig=np.append(msig,prop)
                        msign=np.append(msign,-prop)
                    plt.plot(u2,msig,'r-')
                    plt.plot(u2,msign,'r-')
                    plt.annotate('model variance', xy=(0.40, 0.90), xycoords='axes fraction',color='r')
                    plt.axhline(y=0,c='k')
                if find_jumps:
                    for i in range(1,l_times-1):
                        plt.axvline(x=u_times[i],c='k')
#
            if key in negative:
                plt.title(root+' '+key+' (negative) n='+str(count[key]))
            else:
                plt.title(root+' '+key+ '           n='+str(count[key]))
            tim=epoch.strftime('%Y/%m/%d %H:%M:%S')
            if days:
                plt.xlabel('Days after '+tim)
            else:
                plt.xlabel('Hours after '+tim)
            if plot_raw_data:
                plt.ylabel('Microseconds')
            else:
                plt.ylabel('Residuals (Microseconds)')
            if find_jumps:
                plt.annotate('1st offset '+'{:.3f}'.format(b)+'  (sec/sec) '+ '{:.3e}'.format(m*1e-6)
                    + '  rms '+'{:.3f}'.format(res)+'  (max-min)/rms '+ '{:.1f}'.format(dev),
                    xy=(0.05, 0.95), xycoords='axes fraction')
            elif fit_line:
                plt.annotate('offset '+'{:.3f}'.format(b)+'   (sec/sec) '+ '{:.3e}'.format(m*1e-6)
                    + '   rms '+'{:.3f}'.format(res)+'   (max-min)/rms '+ '{:.1f}'.format(dev),
                    xy=(0.05, 0.95), xycoords='axes fraction')
            if make_plot:
                plt.show()
            if plot_files:
                plot.savefig(root+'_'+key+'.pdf')
