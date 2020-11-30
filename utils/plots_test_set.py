import numpy as np
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, Range1d, Label, BoxAnnotation
from bokeh.layouts import column
from bokeh.models.glyphs import Text
from bokeh.models import Legend
from bokeh.plotting import figure, output_file, save
import pandas as pd
import os
import time

epitope_threshold = 0.75
deepipred_results_dir = f'/home/go96bix/projects/raw_data/test_training_data'
outdir = os.path.join(deepipred_results_dir, 'plots2/')
starttime = time.time()

######## Plots #########
print('\nPlotting.')

##### progress vars ####
filecounter = 1
printlen = 1
total = 1000


########################
class Protein_seq():
	def __init__(self, sequence, score, over_threshold, positions=None):
		self.sequence = sequence
		self.score = score
		self.over_threshold = over_threshold
		if positions == None:
			self.positions = list(range(1, len(self.sequence) + 1))
		else:
			self.positions = positions


def readFasta_extended(file):
	## read fasta file
	header = ""
	seq = ""
	values = []
	with open(file, "r") as infa:
		for index, line in enumerate(infa):
			line = line.strip()
			if index == 0:
				header = line[1:].split("\t")
			elif index == 1:
				seq += line
			elif index == 2:
				pass
			else:
				values = line.split("\t")
	return header, seq, values


def frame_avg(values, frame_extend=2):
	averages = []
	protlen = len(values)
	for pos in range(protlen):
		framelist = []
		for shift in range(-frame_extend, frame_extend + 1, 1):
			if not (pos + shift) < 0 and not (pos + shift) > (protlen - 1):
				framelist.append(float(values[pos + shift]))
		averages.append(sum(framelist) / len(framelist))
	return averages


holydict = {}

for root, dirs, files in os.walk(os.path.join(deepipred_results_dir, "epidope/"), topdown=False):
	for name in files:
		file = os.path.join(root, name)
		df = pd.read_csv(file, sep='\t', index_col=False, skiprows=1)
		letter_arr = df.values[:, 1]
		value_arr = np.array(df.values[:, 2], dtype=np.float)

		score_bool = value_arr > epitope_threshold
		protein = Protein_seq(sequence="".join(letter_arr), score=value_arr, over_threshold=score_bool)
		holydict.update({name[:-(len(".csv"))]: protein})

for geneid in holydict:

	############### progress ###############
	elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - starttime))
	printstring = f'Plotting: {geneid}    File: {filecounter} / {total}   Elapsed time: {elapsed_time}'
	if len(printstring) < printlen:
		print(' ' * printlen, end='\r')
	print(printstring, end='\r')
	printlen = len(printstring)
	filecounter += 1
	#######################################

	# make output dir and create output filename
	if not os.path.exists(outdir + '/plots'):
		os.makedirs(outdir + '/plots')
	out = f'{outdir}/plots/{geneid}.html'
	output_file(out)

	seq = holydict[geneid].sequence
	pos = holydict[geneid].positions
	score = holydict[geneid].score
	flag = holydict[geneid].over_threshold
	# pwa_score = pwa(score, frame_extend = 24)
	protlen = len(seq)

	# create a new plot with a title and axis labels
	p = figure(title=geneid, y_range=(-0.03, 1.03), y_axis_label='Scores', plot_width=1200, plot_height=460,
	           tools='xpan,xwheel_zoom,reset', toolbar_location='above')
	p.min_border_left = 80

	# add a line renderer with legend and line thickness
	l1 = p.line(range(1, protlen + 1), score, line_width=1, color='black', visible=True)
	l2 = p.line(range(1, protlen + 1), ([epitope_threshold] * protlen), line_width=1, color='red', visible=True)

	legend = Legend(items=[('EpiDope', [l1]),
	                       ('epitope_threshold', [l2])])

	p.add_layout(legend, 'right')
	p.xaxis.visible = False
	p.legend.click_policy = "hide"

	p.x_range.bounds = (-50, protlen + 51)

	### plot for sequence
	# symbol based plot stuff

	plot = Plot(title=None, x_range=p.x_range, y_range=Range1d(0, 9), plot_width=1200, plot_height=50, min_border=0,
	            toolbar_location=None)

	y = [1] * protlen
	source = ColumnDataSource(dict(x=list(pos), y=y, text=list(seq)))
	glyph = Text(x="x", y="y", text="text", text_color='black', text_font_size='8pt')
	plot.add_glyph(source, glyph)
	label = Label(x=-80, y=y[1], x_units='screen', y_units='data', text='Sequence', render_mode='css',
	              background_fill_color='white', background_fill_alpha=1.0)
	plot.add_layout(label)

	xaxis = LinearAxis()
	plot.add_layout(xaxis, 'below')
	plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))

	# add predicted epitope boxes
	predicted_epitopes = []
	pred_pos = [i for i, c in enumerate(flag) if c]
	if len(pred_pos) > 1:
		start = pred_pos[0]
		stop = pred_pos[0]
		for i in range(1, len(pred_pos)):
			if pred_pos[i] == stop + 1:
				stop = pred_pos[i]
			else:
				if stop > start:
					predicted_epitopes.append((start, stop))
				start = pred_pos[i]
				stop = pred_pos[i]
		predicted_epitopes.append((start, stop))
	# print(predicted_epitopes)

	for prediction in predicted_epitopes:
		start = prediction[0]
		stop = prediction[1]
		y = np.array([-0.02] * protlen)
		y[start:stop] = 1.02
		p.vbar(x=list(pos), bottom=-0.02, top=y, width=1, alpha=0.2, line_alpha=0, color='darkgreen',
		       legend='predicted_epitopes', visible=True)

	# add known epitope boxes
	header, seq, values = readFasta_extended(
		f'/home/go96bix/projects/raw_data/bepipred_proteins_with_marking/{geneid}.fasta')
	for head in header:
		file_name = head.split("_")
		start, stop = int(file_name[2]) + 1, int(file_name[3]) + 1
		y = np.array([-0.02] * protlen)
		y[start:stop] = 1.02
		if file_name[0].startswith("Negative"):
			p.vbar(x=list(pos), bottom=-0.02, top=y, width=1, alpha=0.2, line_alpha=0, color='darkred',
			       legend='provided_non_epitope', visible=False)
		else:
			p.vbar(x=list(pos), bottom=-0.02, top=y, width=1, alpha=0.2, line_alpha=0, color='blue',
			       legend='provided_epitope', visible=False)

	save(column(p, plot))
'''
	# DeepLoc barplot
	deeploclocations = ['Membrane','Nucleus','Cytoplasm','Extracellular','Mitochondrion','Cell_membrane','Endoplasmic_reticulum','Plastid','Golgi_apparatus','Lysosome/Vacuole','Peroxisome']
	deepplot = figure(x_range=deeploclocations, plot_height=350, title="DeepLoc", toolbar_location=None, tools="")
	deepplot.vbar(x = deeploclocations, top=deeploc_dict[geneid], width = 0.8)
	deepplot.xgrid.grid_line_color = None
	deepplot.xaxis.major_label_orientation = pi/2
	deepplot.y_range.start = 0

	save(column(p,plot,deepplot))
'''
