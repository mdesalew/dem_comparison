
# Set working directory
setwd('D:/R/dem_paper')

# Load packages
library(readxl)
library(ggplot2)  
library(tidyverse)

# Load data
est<-read_excel("discharge.xlsx", sheet = 'est')
eth<-read_excel("discharge.xlsx", sheet = 'eth')
esp<-read_excel("discharge.xlsx", sheet = 'esp')  
usa<-read_excel("discharge.xlsx", sheet = 'usa')

## ESP
esp1<- esp %>%
  gather(Product, values, "Observed":"TanDEM")

esp1$Product <- factor(esp1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

esp1$Date <- as.Date(esp1$Date, format = "%Y/%m/%d")

ggplot()+
  geom_line(esp1, mapping = aes(x=esp1$Date, y=values, group=Product,color=Product))+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("Time (year/month)")+
  theme_grey(base_size = 18)+
  scale_x_date(date_breaks = "2 years", date_labels = "%Y-%m")+
  scale_colour_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))
ggsave("./plots/discharge_SPN.png", width = 10, height= 3, dpi=300)

## EST
est1<- est %>%
  gather(Product, values, "Observed":"TanDEM")

est1$Product <- factor(est1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

est1$Date <- as.Date(est1$Date, format = "%Y/%m/%d")

ggplot()+
  geom_line(est1, mapping = aes(x=est1$Date, y=values, group=Product,color=Product))+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("Time (year/month)")+
  theme_grey(base_size = 18)+
  scale_x_date(date_breaks = "2 years", date_labels = "%Y-%m")+
  scale_colour_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))
ggsave("./plots/discharge_EST.png", width = 10, height= 3, dpi=300)

## ETH
eth1<- eth %>%
  gather(Product, values, "Observed":"TanDEM")

eth1$Product <- factor(eth1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

eth1$Date <- as.Date(eth1$Date, format = "%Y/%m/%d")

ggplot()+
  geom_line(eth1, mapping = aes(x=eth1$Date, y=values, group=Product,color=Product))+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("Time (year/month)")+
  theme_grey(base_size = 18)+
  scale_x_date(date_breaks = "1 years", date_labels = "%Y-%m")+
  scale_colour_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))
ggsave("./plots/discharge_ETH.png", width = 10, height= 3, dpi=300)

## USA
usa1<- usa %>%
  gather(Product, values, "Observed":"TanDEM")

usa1$Product <- factor(usa1$Product, levels = c("Observed","AW3D30","HydroSHEDS","MERIT","NASADEM", "TanDEM"))   # changing the order

usa1$Date <- as.Date(usa1$Date, format = "%Y/%m/%d")

ggplot()+
  geom_line(usa1, mapping = aes(x=usa1$Date, y=values, group=Product,color=Product))+
  ylab(bquote("Discharge"~(m^3~s^-1)))+ 
  xlab("Time (year/month)")+
  theme_grey(base_size = 18)+
  scale_x_date(date_breaks = "2 years", date_labels = "%Y-%m")+
  scale_colour_manual(values = c('dodgerblue3', "magenta2", "goldenrod2","seagreen3","tomato2","bisque4"))
ggsave("./plots/discharge_USA.png", width = 10, height= 3, dpi=300)


