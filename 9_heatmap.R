# Clear workspace and load libraries
rm(list=ls())
library(dplyr);library(RColorBrewer);library(stringr);
library(ggplot2);library(data.table);library(ggforce);
library(extrafont);library(pacman);library(colorRamps)
source("/get_location_metadata.R")
source("/get_cause_metadata.R")

# Set options
options(OutDec="\u0B7")

# Get metadata
loc_meta <- get_location_metadata(location_set_id = 1, gbd_round_id = 6)
cause_meta <- get_cause_metadata(cause_set_id = 3, gbd_round_id = 6)

# Read data
data_dir <- ""
data <- fread(paste0(data_dir, "results.csv"))

# Create display name
data$display_name <- paste(data$intervention, data$cause)

# Rank by location 
data[, rank := frank(predicted_icer_usd_median, ties.method = "first"), by = c("location_id")]

# Merge data with metadata
data <- merge(data, loc_meta[, c("location_id", "lancet_label", "sort_order", "parent_id")], by = "location_id")
setnames(data, "lancet_label", "lancet_label_location")
setnames(data, "sort_order", "sort_order_location")

# Drop SA provinces for resub
data <- data[parent_id != 196]

# Merge with location metadata twice to get lancet label
loc_meta$lowercase_location_name <- tolower(loc_meta$location_name)
data <- merge(data, loc_meta[, c("lowercase_location_name", "lancet_label", "sort_order")],
              by.x = "super_region_name",
              by.y = "lowercase_location_name")
setnames(data, "lancet_label", "lancet_label_superregion")
setnames(data, "sort_order", "sort_order_superregion")

# Intervention display name
data$intervention_name <- paste0(data$intervention,
                                 ifelse(!is.na(data$updated_population_age), paste0(", ", data$updated_population_age), ""))

# Order causes
order <- c("Antental syphilis screening, 0 to 11 m",
           "Chemotherapy for DS-TB",
           "Xpert TB test",
           "BCG vaccine against TB, 0 to 4 y",
           "Preventive therapy for TB, adults",
           "Option B+, women 10-49 y, 0 to 11 m",
           "ART for prevention, adults",
           "ART for prevention, 0 to 9 y",
           "PREP for MSM, adults",
           "PREP for heterosexuals, adults",
           "IPT for pregnant women, 0 to 11 m",
           "IPT for infants, 0 to 11 m",
           "Malaria vaccine, 0 to 4 y",
           "Bed nets")

order <- data.frame(intervention_name = order, intervention_order = seq(1:length(order)))
data_all <- merge(data, order, by = "intervention_name", all.x = TRUE)
data_all <- data_all[order(intervention_order),]
data_all$intervention_name <- factor(data_all$intervention_name, levels = order$intervention_name)
data_all <- data_all[order(lancet_label_location, intervention_order),]

# Intervention display cost 
data_all$display_cost <- ifelse(data_all$predicted_icer_usd_median >= 10000, 
                                formatC(signif(data_all$predicted_icer_usd_median, 3), format = "d", big.mark = " "),
                                signif(data_all$predicted_icer_usd_median, 3))
data_all$display_cost <- paste0("$", data_all$display_cost)
data_all$display_cost[data_all$predicted_icer_usd_median <= 0.5] <- "< $1"

# Get colors
oranges <- c("#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", "#f16913", "#d94801")
greens <- brewer.pal(7, "Greens")
colors <- c(rev(greens), oranges)

# Create heatmaps
heat <- function(data_plot) {
  ggplot(data_plot, 
         aes(x = factor(lancet_label_location, levels = rev(unique(data_plot$lancet_label_location))), 
             y = intervention_name,
             fill = as.factor(rank))) + 
    coord_flip() + 
    facet_col(~lancet_label_superregion, scales = "free_y", space = "free") +
    scale_y_discrete(position = "right", labels = function(x) str_wrap(x, width = 40)) + 
    geom_tile(color = "white", width = 1, height = 1, linewidth = 0.1) + 
    geom_text(label = data_plot$display_cost, size = 2.5, colour = ifelse(data_plot$rank %in% c(1,2,3,13,14), "white", "black")) +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 0),
          axis.title.y = element_text(hjust = 0.5),
          axis.text.y = element_text(angle = 0, hjust = 1),
          legend.position = "none",
          strip.background = element_blank(),
          panel.border = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_blank(),
          plot.margin = unit(c(.25,.75,.5,.1), "in")) + 
    scale_fill_manual(values = colors, na.value = "white") + 
    xlab("Location Names") + 
    ylab("Interventions")
}

# Create heatmaps
heat(data_all[lancet_label_superregion %in% c("Central Europe, eastern Europe, and central Asia",
                                              "Latin America and Caribbean")])

heat(data_all[lancet_label_superregion %in% c("North Africa and Middle East",
                                              "South Asia",
                                              "Southeast Asia, east Asia, and Oceania")])

heat(data_all[lancet_label_superregion %in% c("Sub-Saharan Africa")])

# Save plots
loadfonts()
pdf("heatmap.pdf", width = 8, height = 11, family = 'Shaker 2 Lancet Regular', pointsize = 6)
  plot(heat(data_1))
  plot(heat(data_2))
  plot(heat(data_3))
dev.off()