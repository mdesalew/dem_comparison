# set working directory
setwd('D:/R/dem_paper')

# Load packages
library(readxl)
library(ggplot2)
require(tidyverse)

# Load data
est<-read_excel("discharge.xlsx", sheet = 'est')
eth<-read_excel("discharge.xlsx", sheet = 'eth')
esp<-read_excel("discharge.xlsx", sheet = 'esp')
usa<-read_excel("discharge.xlsx", sheet = 'usa')

### EST
est1 <- est %>%
  gather(Product, value, "Observed":"TanDEM")

est1$Product <- factor(est1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

ggplot(est1, aes(x=Product, y=value, fill= Product)) +
  geom_boxplot()+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("Estonia")+
  theme(axis.title.y = element_text(size = 20))+
  theme(axis.text.x = element_text(size = 20, angle = 25, vjust = 0.5))+
  theme(axis.text.y = element_text(size = 20))+
  theme(legend.position = "none")+
  scale_fill_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))

ggsave("./plots/boxplot_EST.png", width = 8, height= 5, dpi=300)

## ETH
eth1 <- eth %>%
  gather(Product, value, "Observed":"TanDEM")

eth1$Product <- factor(eth1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

ggplot(eth1, aes(x=Product, y=value, fill= Product)) +
  geom_boxplot()+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab('Ethiopia')+
  theme(axis.title.y = element_text(size = 20))+
  theme(axis.text.x = element_text(size = 20, angle = 25, vjust = 0.5))+
  theme(axis.text.y = element_text(size = 20))+
  theme(legend.position = "none")+
  scale_fill_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))

ggsave("./plots/boxplot_ETH.png", width = 8, height= 5, dpi=300)

## ESP
esp1 <- esp %>%
  gather(Product, value, "Observed":"TanDEM")

esp1$Product <- factor(esp1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

ggplot(esp1, aes(x=Product, y=value, fill= Product)) +
  geom_boxplot()+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("Spain")+
  theme(axis.title.y = element_text(size = 20))+
  theme(axis.text.x = element_text(size = 20, angle = 25, vjust = 0.5))+
  theme(axis.text.y = element_text(size = 20))+
  theme(legend.position = "none")+
  scale_fill_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))
  
ggsave("./plots/boxplot_SPN.png", width = 8, height= 5, dpi=300)

## USA
usa1 <- usa %>%
  gather(Product, value, "Observed":"TanDEM")

usa1$Product <- factor(usa1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

ggplot(usa1, aes(x=Product, y=value, fill= Product)) +
  geom_boxplot()+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("USA")+
  theme(axis.title.y = element_text(size = 20))+
  theme(axis.text.x = element_text(size = 20, angle = 25, vjust = 0.5))+
  theme(axis.text.y = element_text(size = 20))+
  theme(legend.position = "none")+
  scale_fill_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))
  
ggsave("./plots/boxplot_USA.png", width = 8, height= 5, dpi=300)


