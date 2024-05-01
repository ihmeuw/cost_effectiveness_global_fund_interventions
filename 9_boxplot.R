rm(list=ls())
# shared functions import
source("/get_location_metadata.R")
source("/get_cause_metadata.R")
source("/get_age_metadata.R")
library(tidyverse);library(reshape2);library(writexl);library(grid);library(gridExtra)
library(ggpubr);library(RColorBrewer);library(scales);library(ggplot2);library(data.table)
library(stringr);library(extrafont);library(tidytext)

# use lancet decimal points to 1 sigfig
options(OutDec="\u0B7")
options(digits=1)
# disable printing results in scientific notation
options(scipen = 100) 

# import lancet font
loadfonts()
f3 = "Shaker 2 Lancet Regular"
#### get the data ####
# read input data
data_dir = ""
data <- fread(paste0(data_dir, 
                     "input_data.csv"))

##### get location metadata
location_meta <- get_location_metadata(location_set_id = 1, gbd_round_id = 6)
age_meta <- get_age_metadata(age_group_set_id = 12, gbd_round_id = 6)

# location names list for ggplot
loc_name_list <- location_meta$location_name
names(loc_name_list) <- location_meta$location_id

# get cause name meta
cause_meta <- get_cause_metadata(cause_set_id = 3, gbd_round_id = 6)

#  get cause metadata
data = merge(data, cause_meta[, c("acause", "lancet_label", "cause_id", "sort_order")], by.x = "cause", by.y = "lancet_label")
setnames(data, "cause", "lancet_label_cause")
setnames(data, "sort_order", "sort_order_cause")

# merge for sort_order and lancet labels
data = merge(data, location_meta[, c("location_id", "lancet_label", "sort_order", "parent_id")], by = "location_id")
setnames(data, "lancet_label", "lancet_label_location")

# dropping SA provinces for resub
data = data[parent_id != 196]

# cause and location lists from data
location_list = unique(data[,location_id], )
cause_list = unique(data[,cause_id], )

# creating a age column for display
data[, display_intervention := intervention]

# don't display all ages if all ages
data[population_age == "all ages", population_age := NA]
data$display_text_boxplot = paste0(data$display_intervention, 
                                   ifelse(!is.na(data$population_age), paste0("\n", data$population_age), ""))

# colors
data$lancet_label_cause=factor(data$lancet_label_cause)
myColors <- brewer.pal(length(unique(data$lancet_label_cause)), "Spectral") # get color for each cause
names(myColors) <- levels(data$lancet_label_cause)

# replace 0 with 0.01 for log transform
value_columns = grep("predicted", colnames(data))
data[, (value_columns) := lapply(.SD, function(x) replace(x, which((x == 0)), 0.01)),
   .SDcols = value_columns]

# get gdp and country data in unique object
location_gdp = unique(data[, c("location_name", 'GDP_2019usd', 'gdp_threshold_pichon', 'lancet_label_location')])
# taking the first occurence (confirmed it's rounding)
location_gdp <- location_gdp[match(unique(location_gdp$location_name), location_gdp$location_name),]
location_gdp = location_gdp[order(lancet_label_location)]
# 
#### sort by predicted_ICER per country---------
out_dir <- paste0("")
pdf(file=paste0(out_dir, "all_countries_leaguetable.pdf"), family = f3, height=8, width = 11, pointsize = 3)

for (i in 1:nrow(location_gdp)) {
    #subset data for country
  subset = data[location_name == location_gdp[i,location_name]]

  #subset data for boxes that are tiny and without colors (lower and upper percentile are the same)
  small_boxes = subset[predicted_icer_usd_75th_per-predicted_icer_usd_25th_per == 0]
  # store country's gbd
  gdp = location_gdp[i, GDP_2019usd]
  gdp_threshold = location_gdp[i, gdp_threshold_pichon]
    
  # get min/max values of scale
  min = min((subset[,predicted_icer_usd_lower]), (subset[,predicted_icer_usd_25th_per]), (0.5*gdp), gdp*gdp_threshold)
  max = max((subset[,predicted_icer_usd_upper]), (subset[,predicted_icer_usd_75th_per]), gdp*gdp_threshold)
  c((subset[,predicted_icer_usd_lower]), (subset[,predicted_icer_usd_25th_per]), (0.5*gdp), gdp*gdp_threshold)
  # create list of axis to subset later, including min as first element below log scale
  axis_vals_master = c(min,c(1,10,50,100,500,1000,5000,10000,50000,100000,500000,1000000,5000000))
  
  # if min and 1 are too lose, then remove 1 to prevent overlap! 
  if ((1-min) < 0.3) {
    axis_vals_master = axis_vals_master[-2]
  }
  
  # subset based on min
  if(min > 1) {
    axis_vals_master = axis_vals_master[-2]
  }
  
  # subset based on max
  length_max = length(axis_vals_master[(axis_vals_master <= max) == TRUE]) + 1
  axis = na.omit(axis_vals_master[1:length_max])
  
  # axis labels, converting 0.01s back to 0, and displaying 1 decimal point
  scale = data.frame(axis = log10(axis))
  scale$label = axis
  scale$label = replace(scale$label, scale$label == 0.01, 0)
  scale$label = format(scale$label, nsmall = 1)
  
  # replace scale labels of 0.0 with 0
  scale$label = replace(scale$label, scale$label == "     0·0", 0)
  
  # order to be used in scale_x_discrete
  subset <- setorder(subset, -predicted_icer_usd_median)
  order <- subset$display_text_boxplot
  names(order) <- subset$predicted_icer_usd_median
  
  # annotation text location
  if (max(subset[round(nrow(subset)*0.75):nrow(subset), predicted_icer_usd_75th_per]) > gdp * gdp_threshold) {
    gdp_threshold_position_y = 0.5
    gdp_threshold_align_y = 0
    gdp_threshold_position_x = -.05
  } else {
    gdp_threshold_position_y = nrow(subset) + 1
    gdp_threshold_align_y = 1
    gdp_threshold_position_x = .15
    
  }
  
  if (max(subset[round(nrow(subset)*0.7):nrow(subset), predicted_icer_usd_75th_per]) > gdp & 
      abs(log10(abs(max(subset[round(nrow(subset)*0.7):nrow(subset), predicted_icer_usd_75th_per]))) - log10(gdp)) > 0.1
  ) {
    gdp_position_y = 0.5
    gdp_align = 0
  } else {
    gdp_position_y = nrow(subset) + 1
    gdp_align = 1
  }

  # plot call
  plot = ggplot() + 
    geom_boxplot(data = subset,
                 lwd = .4,
                 stat = "identity",
                 fatten = .75,
                 show.legend = T, 
                 aes(x = display_text_boxplot,
                     ymin = log10(predicted_icer_usd_lower),
                     lower = log10(predicted_icer_usd_25th_per),
                     middle = log10(predicted_icer_usd_median),
                     upper = log10(predicted_icer_usd_75th_per),
                     ymax = log10(predicted_icer_usd_upper),
                     fill = lancet_label_cause)) +
    # gbd lines
    geom_hline(yintercept = log10(gdp), size = 0.7, color = "#005824", linetype = "dashed") +
    # replace this line with gbd 4 per capita
    geom_hline(yintercept = log10(gdp_threshold * gdp), size = 0.7, color = "#ABDDA4", linetype = "solid") +
    annotate(geom = "text",
             x = gdp_position_y,
             y = log10(gdp) + ifelse(gdp * gdp_threshold > gdp,     
                                     -.05, .17),      # depending on the position of GBD threshold, place label to the left or right of the threshold line.   
             label = c("GDP per capita"),
             angle = 90,
             vjust = 0,
             hjust = gdp_align) +
    annotate(geom = "text",
             x = gdp_threshold_position_y,
             y = log10(gdp_threshold * gdp) + ifelse(gdp_threshold_position_x < 0, 
                                                     gdp_threshold_position_x,
                                                      ifelse((abs(gdp_threshold-1) > 0.3 |
                                                              gdp * gdp_threshold >= gdp), 
                                                           0.17, -0.05)),
             label = c("Country specific GDP threshold"),
             angle = 90, 
             vjust = 0,
             hjust = gdp_threshold_align_y) +
    # use scale created above
    scale_y_continuous(breaks = scale$axis,
                       labels = scale$label,
                       limits = c(min(scale$axis),
                                  max(ceiling(log10(max)), log10(3*gdp)))) + 
    # use axis labesl ordered above
    scale_x_discrete(limits = order) + 
    # title and axis text
    ggtitle(paste0("Interventions for HIV/AIDS, malaria, syphilis, and tuberculosis ranked by incremental cost-effectiveness ratio (ICER) in ",
                 location_gdp[i,lancet_label_location], " in 2019")) + 
    xlab("Cause, intervention and age group") + 
    ylab("Cost per disability-adjusted life-year (DALY) averted in 2019 US$") +
    theme_bw() + 
    # axis.text.y bolds x axis text if status == eligible. 
    theme(axis.text.y = element_text(face = ifelse(subset$Status == "eligible", "bold.italic", "plain"), lineheight = unit(.9, 'mm')),
          axis.ticks = element_blank(),
          panel.border = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.background = element_blank(),
          legend.position = 'none',
          legend.spacing.y = unit(0.05, 'in'),
          plot.margin=unit(c(0.1,.5,0.1,.5),"in"),
          axis.text.x = element_text(angle = 45, vjust = 0.8), # angle y axis labels
          plot.title = element_text(hjust = 0.9)) +
    coord_flip() +
    # use colors created above so causes have consistent colors
    scale_fill_manual(values = myColors) + 
    # adding text for interventions that are hard to spot (boxes with the same min and max)
    geom_text(data = small_boxes,    
              aes(x = factor(small_boxes$display_text_boxplot,
                             reorder(small_boxes$display_text_boxplot, -small_boxes$predicted_icer_usd)),
                  y = log10(predicted_icer_usd_upper) + 0.1,
                  label = display_text_boxplot),
              size = 2.5,
              hjust = 0) 
  # only annotate if funding is ineligible intervention exists in country
  if (nrow(subset[Status != "eligible"]) != 0) {
    plot <- annotate_figure(plot,
                            bottom = text_grob("Bolded interventions are eligible for support from The Global Fund.",
                                               x = 0.18,
                                               y = 3.0,
                                               size = 9))
  }

  # save plots
  plot(plot)
  
  print(paste0("done with:", location_gdp[i, location_name]))
}
dev.off()


### figure 3 -----
pdf(file=paste0(out_dir, "fig_3_",version_id, ".pdf"), family = f3, height=8, width = 11, pointsize = 3)
locs = c('India', 'Indonesia', 'Nigeria', 'Peru', 'Sudan', 'Ukraine')
fig2_locs = location_gdp[lancet_label_location %in% locs,]
for (i in 1:nrow(fig2_locs)) {
  
  # uncomment the 2 lines below and set test_runs to the row index of the desired country.
  # for (i in 1:10) {
  #subset data for country
  subset = data[location_name == fig2_locs[i,location_name]]
  
  #subset data for boxes that are tiny and without colors (lower and upper percentile are the same)
  small_boxes = subset[predicted_icer_usd_75th_per-predicted_icer_usd_25th_per == 0]
  # store country's gbd
  gdp = fig2_locs[i, GDP_2019usd]
  gdp_threshold = fig2_locs[i, gdp_threshold_pichon]
  
  # get min/max values of scale
  min = min((subset[,predicted_icer_usd_lower]), (subset[,predicted_icer_usd_25th_per]), (0.5*gdp), gdp*gdp_threshold)
  max = max((subset[,predicted_icer_usd_upper]), (subset[,predicted_icer_usd_75th_per]), gdp*gdp_threshold)
  # create list of axis to subset later, including min as first element below log scale
  axis_vals_master = c(min,c(1,10,50,100,500,1000,5000,10000,50000,100000,500000,1000000,5000000))
  
  # if min and 1 are too lose, then remove 1 to prevent overlap! 
  if ((1-min) < 0.3) {
    axis_vals_master = axis_vals_master[-2]
  }
  
  # subset based on min
  if(min > 1) {
    axis_vals_master = axis_vals_master[-2]
  }
  
  # subset based on max
  length_max = length(axis_vals_master[(axis_vals_master <= max) == TRUE]) + 1
  axis = na.omit(axis_vals_master[1:length_max])
  
  # axis labels, converting 0.01s back to 0, and displaying 1 decimal point
  scale = data.frame(axis = log10(axis))
  scale$label = axis
  scale$label = replace(scale$label, scale$label == 0.01, 0)
  scale$label = format(scale$label, nsmall = 1)
  
  # replace scale labels of 0.0 with 0
  scale$label = replace(scale$label, scale$label == "     0·0", 0)
  
  # order to be used in scale_x_discrete
  subset <- setorder(subset, -predicted_icer_usd_median)
  order <- subset$display_text_boxplot
  names(order) <- subset$predicted_icer_usd_median
  
  # annotation text location
  if (max(subset[round(nrow(subset)*0.75):nrow(subset), predicted_icer_usd_75th_per]) > gdp * gdp_threshold) {
    gdp_threshold_position_y = 0.5
    gdp_threshold_align_y = 0
    gdp_threshold_position_x = -.05
  } else {
    gdp_threshold_position_y = nrow(subset) + 1
    gdp_threshold_align_y = 1
    gdp_threshold_position_x = .15
    
  }
  
  if (max(subset[round(nrow(subset)*0.7):nrow(subset), predicted_icer_usd_75th_per]) > gdp & 
      abs(log10(abs(max(subset[round(nrow(subset)*0.7):nrow(subset), predicted_icer_usd_75th_per]))) - log10(gdp)) > 0.1
  ) {
    gdp_position_y = 0.5
    gdp_align = 0
  } else {
    gdp_position_y = nrow(subset) + 1
    gdp_align = 1
  }
  
  # plot call
  plot = ggplot() + 
    geom_boxplot(data = subset,
                 lwd = .4,
                 stat = "identity",
                 fatten = .75,
                 show.legend = T, 
                 aes(x = display_text_boxplot,
                     ymin = log10(predicted_icer_usd_lower),
                     lower = log10(predicted_icer_usd_25th_per),
                     middle = log10(predicted_icer_usd_median),
                     upper = log10(predicted_icer_usd_75th_per),
                     ymax = log10(predicted_icer_usd_upper),
                     fill = lancet_label_cause)) +
    # gbd lines
    geom_hline(yintercept = log10(gdp), size = 0.7, color = "#005824", linetype = "dashed") +
    geom_hline(yintercept = log10(gdp_threshold * gdp), size = 0.7, color = "#ABDDA4", linetype = "solid") +
    annotate(geom = "text",
             x = gdp_position_y,
             y = log10(gdp) + ifelse(gdp * gdp_threshold > gdp,     
                                     -.05, .17),      # depending on the position of GBD threshold, place label to the left or right of the threshold line.   
             label = c("GDP per capita"),
             angle = 90,
             vjust = 0,
             hjust = gdp_align) +
    annotate(geom = "text",
             x = gdp_threshold_position_y,
             y = log10(gdp_threshold * gdp) + ifelse(gdp_threshold_position_x < 0, 
                                                     gdp_threshold_position_x,
                                                     ifelse((abs(gdp_threshold-1) > 0.3 |
                                                               gdp * gdp_threshold >= gdp), 
                                                            0.17, -0.05)),
             label = c("Country specific GDP threshold"),
             angle = 90, 
             vjust = 0,
             hjust = gdp_threshold_align_y) +
    # use scale created above
    scale_y_continuous(breaks = scale$axis,
                       labels = scale$label,
                       limits = c(min(scale$axis),
                                  max(ceiling(log10(max)), log10(3*gdp)))) + 
    # use axis labesl ordered above
    scale_x_discrete(limits = order) + 
    # title and axis text
    ggtitle(paste0("Figure 3", letters[i],
                   ". ", fig2_locs[i,lancet_label_location])) + 
    xlab("Cause, intervention and age group") + 
    ylab("Cost per disability-adjusted life-year (DALY) averted in 2019 US$") +
    theme_bw() + 
    # axis.text.y bolds x axis text if status == eligible. 
    theme(axis.text.y = element_text(face = ifelse(subset$Status == "eligible", "bold.italic", "plain"), lineheight = unit(.9, 'mm')),
          axis.ticks = element_blank(),
          panel.border = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.background = element_blank(),
          legend.position = 'none',
          legend.spacing.y = unit(0.05, 'in'),
          plot.margin=unit(c(0.1,.5,0.1,.5),"in"),
          axis.text.x = element_text(angle = 45, vjust = 0.8), # angle y axis labels
          plot.title = element_text(hjust = 0.5)) +
    coord_flip() +
    # use colors created above so causes have consistent colors
    scale_fill_manual(values = myColors) + 
    # adding text for interventions that are hard to spot (boxes with the same min and max)
    geom_text(data = small_boxes,    
              aes(x = factor(small_boxes$display_text_boxplot,
                             reorder(small_boxes$display_text_boxplot, -small_boxes$predicted_icer_usd)),
                  y = log10(predicted_icer_usd_upper) + 0.1,
                  label = display_text_boxplot),
              size = 2.5,
              hjust = 0) 
  # only annotate if funding is ineligible intervention exists in country
  if (nrow(subset[Status != "eligible"]) != 0) {
    plot <- annotate_figure(plot,
                            bottom = text_grob("Bolded interventions are eligible for support from The Global Fund.",
                                               x = 0.18,
                                               y = 3.0,
                                               size = 9))
  }
  
  # save plots
  plot(plot)
  
  print(paste0("done with:", fig2_locs[i, location_name]))
}
dev.off()