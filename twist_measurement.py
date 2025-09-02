#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EUV Flux Rope Twist Measurement Tool with Magnetogram included to better measure it.

This tool combines EUV and magnetogram data to measure flux rope twist parameters
while simultaneously visualizing the magnetic field context.

Features:
- Multi-spacecraft support (SDO/AIA, STEREO/EUVI, SolO/EUI)
- Synchronized EUV and magnetogram visualization
- Automatic magnetogram reprojection to match EUV perspective
- Interactive line thickness control
- Physical units (km) calculations
- Optional FITS file downloading
- ROI selection
- Measurement coordinate saving

Controls:
'b': Blue line mode (Length L measurement)
'r': Red line mode (Width a measurement)
Click and drag: Draw lines (appears on both EUV and magnetogram)
'c': Clear current measurement
'm': Store measurement and start new
'q': Quit and save results

Required Input:
- EUV FITS file (AIA 171/193/211/304, EUVI 171/195/284/304, or EUI 174/304)
- HMI magnetogram FITS file
- Output directory path

Output:
- Measurement results (L, a, T_TDm, T_obs)
- Coordinate files with heliographic positions
- Visualization images showing both EUV and magnetogram context
"""

# Imports
import cv2
import numpy as np
import os
import sys
from datetime import datetime, timedelta 
import sunpy.map
from sunpy.coordinates import frames
from sunpy.coordinates.frames import Helioprojective
import astropy.units as u
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch
import warnings
from astropy.utils.exceptions import AstropyWarning
from sunpy.net import Fido, attrs as a
import sunpy_soar
import matplotlib.colors as colors
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tkinter import filedialog

# Suppress warnings
warnings.filterwarnings('ignore', category=AstropyWarning)

plt.rcParams['font.family'] = 'Times New Roman'

class TwistMeasurementWithMag:
    """Handles simultaneous EUV and magnetogram analysis for flux rope twist measurements."""
    
    def __init__(self, euv_path=None, mag_path=None, output_path=None):
        """Initialize measurement tool."""
        # Basic initialization
        self.euv_path = euv_path
        self.mag_path = mag_path
        self.base_output_path = output_path or os.getcwd()
        self.output_dir = None  # Will be set in load_data
        
        # Display parameters
        self.display_resolution = 1024  # Maximum width/height for display
        self.downsample_factor = 1.0    # Will be updated based on image size
        
        # Drawing parameters
        self.line_thickness = 2
        self.drawing = False
        self.mode = 'blue'
        self.measurements = []
        self.current_lines = {'blue': [], 'red': []}
        self.current_points = []
        self.all_measurements_lines = []
        
        # Load data or enter download mode
        if euv_path is None or mag_path is None:
            self.download_mode()
        else:
            self.load_data()
        
        # Create window and set up interaction after data is loaded
        cv2.namedWindow('Measurement View', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Line Thickness', 'Measurement View', 2, 20, self.on_thickness_change)
        cv2.setMouseCallback('Measurement View', self.draw_line)
    
    def download_mode(self):
        """Interactive download interface for EUV and HMI data."""
        print("\nInitializing download mode...")
        
        if self.euv_path:
            print(f"\nUsing provided EUV file")
            try:
                # Get time range from EUV file
                print("Loading provided EUV file...")
                temp_map = sunpy.map.Map(self.euv_path)
                t = temp_map.date
                t1 = t - timedelta(minutes=1)
                t2 = t + timedelta(minutes=3)
                time_range = [t1.strftime("%Y-%m-%d %H:%M"), 
                             t2.strftime("%Y-%m-%d %H:%M")]
                
                # Setup download directory
                download_dir = os.path.join(self.base_output_path, 'downloaded_fits')
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)
                
                # Only download HMI
                print("Downloading matching HMI magnetogram...")
                _, self.mag_path = self.download_euv_and_mag(time_range, download_dir=download_dir)
                self.load_data()
                
            except Exception as e:
                print(f"HMI download failed: {e}")
                raise ValueError(f"HMI download failed: {e}")
        
        else:
            print("\nFITS File Download Interface")
            print("----------------------------")
            print("Available instruments:")
            print("1. SDO/AIA (171Å, 193Å, 211Å, 304Å)")
            print("2. STEREO-A/EUVI (171Å, 195Å, 284Å, 304Å)")
            print("3. Solar Orbiter/EUI (174Å, 304Å)")
            
            instrument_choice = input("Select instrument (1-3): ")
            
            wavelength_options = {
                '1': {'1': 171, '2': 193, '3': 211, '4': 304},
                '2': {'1': 171, '2': 195, '3': 284, '4': 304},
                '3': {'1': 174, '2': 304}
            }
            
            if instrument_choice in wavelength_options:
                print("\nAvailable wavelengths:")
                for key, wave in wavelength_options[instrument_choice].items():
                    print(f"{key}. {wave}Å")
                
                wave_choice = input("Select wavelength: ")
                selected_wave = wavelength_options[instrument_choice].get(wave_choice)
                
                instrument_map = {
                    '1': 'aia',
                    '2': 'secchi',
                    '3': 'eui'
                }
                
                if selected_wave:
                    inst = instrument_map[instrument_choice]
                    wave = selected_wave * u.angstrom
                    
                    print("\nEnter reference time (format: YYYY-MM-DD HH:MM)")
                    time_point = input("Time: ").strip()
                    
                    try:
                        t = datetime.strptime(time_point, "%Y-%m-%d %H:%M")
                        t1 = t - timedelta(minutes=1)
                        t2 = t + timedelta(minutes=3)
                        time_range = [t1.strftime("%Y-%m-%d %H:%M"), 
                                    t2.strftime("%Y-%m-%d %H:%M")]
                        
                        download_dir = os.path.join(self.base_output_path, 'downloaded_fits')
                        if not os.path.exists(download_dir):
                            os.makedirs(download_dir)
                        
                        print(f"Downloading data...")
                        self.euv_path, self.mag_path = self.download_euv_and_mag(
                            time_range,
                            instrument=inst,
                            wavelength=wave,
                            download_dir=download_dir
                        )
                        
                        self.load_data()
                        
                    except Exception as e:
                        print(f"Download failed: {e}")
                        raise ValueError(f"Download failed: {e}")
                else:
                    raise ValueError("Invalid wavelength selection")
            else:
                raise ValueError("Invalid instrument selection")

    def download_euv_and_mag(self, time_range, instrument='aia', wavelength=171*u.angstrom, download_dir=None):
        """Download EUV and HMI data using Fido"""
        print(f"\nSearching data for:")
        print(f"Time range: {time_range}")
        print(f"Instrument: {instrument}")
        print(f"Wavelength: {wavelength}")
                
        try:
            # Handle provided EUV path first
            if self.euv_path:
                print("Using provided EUV file")
                euv_path = self.euv_path
                
                # If HMI path provided, return both
                if self.mag_path:
                    return euv_path, self.mag_path
                    
                # Only download HMI if needed
                print(f"\nSearching for HMI data:")
                print(f"Time range: {time_range}")
                
                mag_query = Fido.search(
                    a.Time(time_range[0], time_range[1]),
                    a.Instrument("hmi"),
                    a.Physobs("LOS_magnetic_field")
                )
                
                print(f"Found {len(mag_query)} magnetogram results")
                if len(mag_query) > 0:
                    print("Downloading HMI magnetogram...")
                    mag_path = Fido.fetch(mag_query[0][0], path=download_dir)[0]
                    print(f"Downloaded magnetogram file: {os.path.basename(mag_path)}")
                    return euv_path, mag_path
                else:
                    raise ValueError("No HMI data found")
                    
            # Original download logic for when no EUV path provided
            else:
                # EUI specific download
                if instrument == 'eui':
                    import sunpy_soar
                    # Select product name based on wavelength
                    if wavelength.value == 174:
                        product = "EUI-FSI174-IMAGE"
                    elif wavelength.value == 304:
                        product = "EUI-FSI304-IMAGE"
                    else:
                        raise ValueError(f"Unsupported EUI wavelength: {wavelength}")
                        
                    print("Searching for EUI data...")
                    euv_query = Fido.search(
                        a.Time(time_range[0], time_range[1]) &
                        a.Instrument("EUI") &
                        a.Level(2) &
                        a.soar.Product(product)
                    )
                # AIA/SECCHI download
                else:
                    print(f"Searching for {instrument} data...")
                    euv_query = Fido.search(
                        a.Time(time_range[0], time_range[1]),
                        a.Instrument(instrument),
                        a.Wavelength(wavelength)
                    )
                    
                print(f"Found {len(euv_query)} EUV results")
                if len(euv_query) > 0:
                    print("Downloading EUV data...")
                    euv_path = Fido.fetch(euv_query[0], path=download_dir)[0]
                    print(f"Downloaded EUV file: {os.path.basename(euv_path)}")
                else:
                    raise ValueError("No EUV data found")
                    
                # Use existing HMI path if provided
                if self.mag_path:
                    return euv_path, self.mag_path
                    
                # Download HMI if no path provided
                print("Searching for HMI data...")
                mag_query = Fido.search(
                    a.Time(time_range[0], time_range[1]),
                    a.Instrument("hmi"),
                    a.Physobs("LOS_magnetic_field")
                )
                
                print(f"Found {len(mag_query)} magnetogram results")
                if len(mag_query) > 0:
                    print("Downloading HMI magnetogram...")
                    mag_path = Fido.fetch(mag_query[0][0], path=download_dir)[0]
                    print(f"Downloaded magnetogram file: {os.path.basename(mag_path)}")
                    return euv_path, mag_path
                else:
                    raise ValueError("No HMI data found")
                    
        except Exception as e:
            print(f"Error during download: {e}")
            raise ValueError(f"Download failed: {str(e)}")

    def load_data(self):
        """Load and prepare EUV and magnetogram data."""
        try:
            print("Loading and processing FITS files...")
            
            # Load EUV map
            self.euv_map = sunpy.map.Map(self.euv_path)
            self.observatory = self.euv_map.observatory
            self.wavelength = self.euv_map.wavelength
            self.timestamp = self.euv_map.date
            
            # Load magnetogram and store original time
            mag_map_raw = sunpy.map.Map(self.mag_path)
            self.mag_time = mag_map_raw.date  # Store original time
            
            print("Reprojecting magnetogram...")
            
            # Reproject magnetogram to match EUV perspective
            out_shape = self.euv_map.data.shape
            mag_reproj = reproject_interp((mag_map_raw.data, mag_map_raw.wcs),
                                        self.euv_map.wcs, 
                                        shape_out=out_shape)
            
            # Store reprojected magnetogram
            self.mag_map = sunpy.map.Map(mag_reproj[0], self.euv_map.wcs)
            self.mag_data = mag_reproj[0]  # Add this line to store the raw data
            
            # Setup output directory
            self.output_dir = os.path.join(self.base_output_path, 
                                         f'twist_measurements_{self.observatory}')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Prepare scientific data
            self.scientific_euv_data = self.euv_map.data
            self.scientific_mag_data = self.mag_map.data
            
            # Prepare display images
            print("Creating display images...")
            self.prepare_display_images()
            
            # Select ROI with improved interface
            print("Select region of interest...")
            self.select_roi()
            
            print(f"Data loaded successfully")
            print(f"EUV time: {self.timestamp}")
            print(f"HMI time: {self.mag_time}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise ValueError(f"Error loading data: {e}")
    
    def prepare_display_images(self):
        """Prepare display images using SunPy's visualization tools at full resolution."""
        try:
            # Store original shape for reference
            self.original_shape = self.euv_map.data.shape
            print(f"Using full original resolution: {self.original_shape[0]}x{self.original_shape[1]}")
            
            # Always use full resolution - no downsampling
            self.downsample_factor = 1.0
            
            # Create temporary matplotlib figures with SunPy maps at full resolution
            # EUV Image with proper colormap
            fig_euv = plt.figure(figsize=(self.original_shape[1]/100, self.original_shape[0]/100), dpi=100)
            ax_euv = fig_euv.add_subplot(111, projection=self.euv_map)
            
            # Select clip interval based on instrument
            if 'Solar Orbiter' in str(self.observatory) or 'SOLO' in str(self.observatory):
                # EUI tends to be very bright, use more aggressive clipping
                self.euv_map.plot(axes=ax_euv, clip_interval=(1, 99.5)*u.percent, title=False)
            else:
                # Standard clipping for other instruments
                self.euv_map.plot(axes=ax_euv, clip_interval=(1, 99.9)*u.percent, title=False)
            
            ax_euv.set_axis_off()
            fig_euv.tight_layout(pad=0)
            
            # Convert matplotlib figure to CV2 image
            canvas = FigureCanvas(fig_euv)
            canvas.draw()
            euv_array = np.array(canvas.renderer.buffer_rgba())
            self.euv_display = cv2.cvtColor(euv_array, cv2.COLOR_RGBA2BGR)
            plt.close(fig_euv)
            
            # Magnetogram with proper colormap
            fig_mag = plt.figure(figsize=(self.original_shape[1]/100, self.original_shape[0]/100), dpi=100)
            ax_mag = fig_mag.add_subplot(111, projection=self.mag_map)
            self.mag_map.plot(axes=ax_mag, cmap='RdBu_r', vmin=-500, vmax=500, title=False)
            ax_mag.set_axis_off()
            fig_mag.tight_layout(pad=0)
            
            # Convert matplotlib figure to CV2 image
            canvas = FigureCanvas(fig_mag)
            canvas.draw()
            mag_array = np.array(canvas.renderer.buffer_rgba())
            self.mag_display = cv2.cvtColor(mag_array, cv2.COLOR_RGBA2BGR)
            plt.close(fig_mag)
            
            # Create combined display
            self.combined_display = np.hstack((self.euv_display, self.mag_display))
            
            print(f"Display images created at full original resolution")
            
        except Exception as e:
            print(f"Error preparing display images: {e}")
            print("Falling back to simple visualization...")
            self.fallback_visualization()

    def fallback_visualization(self):
        """Create simple display images as fallback."""
        try:
            # Calculate display size
            shape = self.euv_map.data.shape
            aspect_ratio = shape[1] / shape[0]
            
            if max(shape) > self.display_resolution:
                if shape[0] > shape[1]:
                    new_h = self.display_resolution
                    new_w = int(new_h * aspect_ratio)
                else:
                    new_w = self.display_resolution
                    new_h = int(new_w / aspect_ratio)
                self.downsample_factor = shape[0] / new_h
            else:
                new_h, new_w = shape
                self.downsample_factor = 1.0
            
            # Simple EUV scaling
            euv_data = self.euv_map.data.copy()
            p_low, p_high = np.percentile(euv_data[np.isfinite(euv_data)], (1, 99.9))
            euv_scaled = np.clip((euv_data - p_low) / (p_high - p_low), 0, 1) * 255
            euv_scaled = np.flipud(euv_scaled.astype(np.uint8))
            
            if self.downsample_factor != 1.0:
                euv_scaled = cv2.resize(euv_scaled, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Apply a green-yellow colormap for EUV
            euv_colored = np.zeros((euv_scaled.shape[0], euv_scaled.shape[1], 3), dtype=np.uint8)
            euv_colored[..., 0] = np.minimum(euv_scaled * 0.2, 255)  # Blue channel (minimal)
            euv_colored[..., 1] = euv_scaled  # Green channel (strong)
            euv_colored[..., 2] = np.minimum(euv_scaled * 0.7, 255)  # Red channel (moderate)
            
            # Simple magnetogram scaling
            mag_data = self.mag_map.data.copy()
            mag_max = 500
            mag_norm = np.clip(mag_data / mag_max, -1, 1)
            
            # Create RdBu colormap for magnetogram
            mag_colored = np.zeros((mag_data.shape[0], mag_data.shape[1], 3), dtype=np.uint8)
            mag_colored[..., 0] = np.where(mag_norm < 0, 0, np.minimum(255 * abs(mag_norm), 255)).astype(np.uint8)  # Red
            mag_colored[..., 1] = np.where(mag_norm == 0, 255, 0).astype(np.uint8)  # Green for zero
            mag_colored[..., 2] = np.where(mag_norm > 0, 0, np.minimum(255 * abs(mag_norm), 255)).astype(np.uint8)  # Blue
            
            # Flip for correct orientation
            mag_colored = np.flipud(mag_colored)
            
            if self.downsample_factor != 1.0:
                mag_colored = cv2.resize(mag_colored, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Set display images
            self.euv_display = euv_colored
            self.mag_display = mag_colored
            
            # Combine displays
            self.combined_display = np.hstack((self.euv_display, self.mag_display))
            print(f"Fallback images created at {new_w}x{new_h} resolution")
            
        except Exception as e:
            print(f"Even fallback visualization failed: {e}")
            # Create blank images as last resort
            self.euv_display = np.zeros((512, 512, 3), dtype=np.uint8)
            self.mag_display = np.zeros((512, 512, 3), dtype=np.uint8)
            self.combined_display = np.hstack((self.euv_display, self.mag_display))
    
    def select_roi(self):
        """Interactive region of interest selection."""
        try:
            print("Select region of interest (drag to select, press ENTER when done)")
            
            # Create a window for ROI selection
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
            
            # Get the display dimensions and set a reasonable window size
            img_h, img_w = self.combined_display.shape[:2]
            window_w = min(1600, img_w)
            window_h = min(1200, img_h)
            cv2.resizeWindow("Select ROI", window_w, window_h)
            
            # Show the image
            cv2.imshow("Select ROI", self.combined_display)
            cv2.waitKey(100)  # Make sure the window appears
            
            roi = cv2.selectROI("Select ROI", self.combined_display, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")
            
            x, y, w, h = roi
            
            # Check if selection is valid (has reasonable size)
            if w > 20 and h > 20:
                # Make sure not to extend into magnetogram side
                euv_width = self.combined_display.shape[1] // 2
                if x + w > euv_width:
                    w = euv_width - x
                    
                self.roi_coords = {'x': x, 'y': y, 'width': w, 'height': h}
                print(f"ROI selected: x={x}, y={y}, width={w}, height={h}")
            else:
                # If selection is too small or cancelled, use default
                half_width = self.combined_display.shape[1] // 2
                self.roi_coords = {'x': 0, 'y': 0, 'width': half_width, 'height': self.combined_display.shape[0]}
                print("Using full EUV image (ROI selection too small or cancelled)")
            
            # Apply the ROI selection
            self.apply_roi()
            
        except Exception as e:
            print(f"Error in ROI selection: {e}")
            # Fall back to full EUV image
            half_width = self.combined_display.shape[1] // 2
            self.roi_coords = {'x': 0, 'y': 0, 'width': half_width, 'height': self.combined_display.shape[0]}
            print("Using full EUV image due to error")
            self.apply_roi()
    
    def apply_roi(self):
        """Apply ROI to both images and create high-resolution submaps."""
        try:
            # Get ROI from display coordinates
            x = self.roi_coords['x']
            y = self.roi_coords['y']
            w = self.roi_coords['width']
            h = self.roi_coords['height']
            
            # Ensure coordinates are valid
            if w <= 0 or h <= 0:
                raise ValueError("Invalid ROI dimensions")
            
            # Check bounds against image dimensions
            h_max, w_max = self.euv_display.shape[:2]
            if x + w > w_max:
                w = w_max - x
            if y + h > h_max:
                h = h_max - y
                    
            print(f"Applying ROI in display coordinates: x={x}, y={y}, width={w}, height={h}")
            
            # Crop display images
            self.euv_display = self.euv_display[y:y+h, x:x+w]
            self.mag_display = self.mag_display[y:y+h, x:x+w]
            
            # Update combined display
            self.combined_display = np.hstack((self.euv_display, self.mag_display))
            self.original_combined = self.combined_display.copy()
            
            # Convert from display coordinates to original image coordinates
            orig_x = int(x * self.downsample_factor)
            orig_y = int(y * self.downsample_factor)
            orig_w = int(w * self.downsample_factor)
            orig_h = int(h * self.downsample_factor)
            
            # Store original coordinates for reference
            self.orig_roi = {'x': orig_x, 'y': orig_y, 'width': orig_w, 'height': orig_h}
            
            print(f"Original image ROI: x={orig_x}, y={orig_y}, width={orig_w}, height={orig_h}")
            print(f"Original image dimensions: {self.euv_map.data.shape}")
            
            # DIAGNOSTIC: Try both approaches to creating submaps
            
            # 1. Use slicing to extract data directly (guaranteed to work)
            self.euv_roi_data = self.euv_map.data[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w].copy()
            self.mag_roi_data = self.mag_data[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w].copy()
            
            print(f"Extracted ROI data shape: {self.euv_roi_data.shape}")
            
            # 2. Try SunPy's submap approach
            try:
                # Convert from pixel to world coordinates
                print("Converting pixel coordinates to world coordinates...")
                bottom_left = self.euv_map.pixel_to_world(
                    orig_x * u.pixel, 
                    (self.euv_map.data.shape[0] - (orig_y + orig_h)) * u.pixel
                )
                top_right = self.euv_map.pixel_to_world(
                    (orig_x + orig_w) * u.pixel, 
                    (self.euv_map.data.shape[0] - orig_y) * u.pixel
                )
                
                print(f"Bottom left world coord: {bottom_left}")
                print(f"Top right world coord: {top_right}")
                
                # Create submaps using world coordinates
                print("Creating submaps using world coordinates...")
                self.euv_submap = self.euv_map.submap(bottom_left, top_right)
                self.mag_submap = self.mag_map.submap(bottom_left, top_right)
                
                print(f"Submap dimensions: {self.euv_submap.data.shape}")
                
                # CRITICAL CHECK: Compare submap to direct extraction
                submap_similar = np.allclose(self.euv_roi_data, self.euv_submap.data, rtol=0.1)
                print(f"Submap similar to direct extraction: {submap_similar}")
                
                if not submap_similar:
                    print("WARNING: Submap data differs significantly from direct extraction!")
                    print(f"Submap min/max: {self.euv_submap.data.min()}/{self.euv_submap.data.max()}")
                    print(f"Direct extraction min/max: {self.euv_roi_data.min()}/{self.euv_roi_data.max()}")
                
                # If submap is entire map, something went wrong
                if self.euv_submap.data.shape == self.euv_map.data.shape:
                    print("ERROR: Submap has same dimensions as original map!")
                    raise ValueError("Submap creation failed")
                    
            except Exception as submap_error:
                print(f"Error creating submap: {submap_error}")
                print("Will use direct data extraction instead of submaps")
                self.use_direct_extraction = True
                
        except Exception as e:
            print(f"Error applying ROI: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to full image
            self.use_direct_extraction = True
            h, w = self.euv_display.shape[:2]
            self.roi_coords = {'x': 0, 'y': 0, 'width': w, 'height': h}
            self.combined_display = np.hstack((self.euv_display, self.mag_display))
            self.original_combined = self.combined_display.copy()
        

    def draw_line(self, event, x, y, flags, param):
        """Handle mouse events for line drawing."""
        euv_width = self.euv_display.shape[1]
        
        if x >= euv_width:  # Ignore clicks on magnetogram side
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_points = [(x, y)]
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.combined_display = self.original_combined.copy()
            self.draw_saved_lines()
            
            color = (255,0,0) if self.mode == 'blue' else (0,0,255)
            cv2.line(self.combined_display, 
                    self.current_points[0], 
                    (x, y),
                    color,
                    self.line_thickness)
            cv2.line(self.combined_display,
                    (self.current_points[0][0] + euv_width, self.current_points[0][1]),
                    (x + euv_width, y),
                    color,
                    self.line_thickness)
                    
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_points.append((x, y))
            self.current_lines[self.mode].append(self.current_points)
            self.draw_saved_lines()

    def draw_saved_lines(self):
        """Redraw all saved lines on both images."""
        euv_width = self.euv_display.shape[1]
        for color in ['blue', 'red']:
            for line in self.current_lines[color]:
                # Check if line has valid points
                if len(line) >= 2:
                    cv2.line(self.combined_display, 
                            line[0], line[1],
                            (255,0,0) if color == 'blue' else (0,0,255),
                            self.line_thickness)
                    cv2.line(self.combined_display,
                            (line[0][0] + euv_width, line[0][1]),
                            (line[1][0] + euv_width, line[1][1]),
                            (255,0,0) if color == 'blue' else (0,0,255),
                            self.line_thickness)

    def on_thickness_change(self, value):
        """Handle line thickness trackbar changes."""
        self.line_thickness = value
        self.combined_display = self.original_combined.copy()
        self.draw_saved_lines()

    def reset_measurement(self):
        """Clear current measurement."""
        self.combined_display = self.original_combined.copy()
        self.current_lines = {'blue': [], 'red': []}
        self.current_points = []
        print("Measurement cleared")

    def calculate_measurements(self):
        """Calculate twist parameters based on current measurements."""
        try:
            # Basic calculations
            rsun_arcsec = self.euv_map.rsun_obs.value
            km_per_arcsec = 695700 / rsun_arcsec
            pixel_scale = self.euv_map.scale[0].value
            pixel_uncertainty = pixel_scale * km_per_arcsec
            
            # Adjust for ROI and downsampling
            scale_factor = self.downsample_factor
            
            # Length calculations with scale correction
            blue_length = sum([np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * scale_factor
                             for p1, p2 in self.current_lines['blue']])
            red_length = sum([np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * scale_factor
                            for p1, p2 in self.current_lines['red']])
            
            L_km = blue_length * pixel_scale * km_per_arcsec
            a_km = red_length * pixel_scale * km_per_arcsec
            
            # Standard deviations if multiple measurements exist
            if len(self.measurements) > 0:
                L_std = np.std([m[0] for m in self.measurements])
                a_std = np.std([m[1] for m in self.measurements])
                # Direct twist std
                T_TDm_std = np.std([m[2] for m in self.measurements])
                T_obs_std = np.std([m[3] for m in self.measurements])
            else:
                L_std = a_std = T_TDm_std = T_obs_std = 0
            
            # Total length uncertainties
            L_error = np.sqrt(pixel_uncertainty**2 + L_std**2)
            a_error = np.sqrt(pixel_uncertainty**2 + a_std**2)
            
            # Ratio and propagated uncertainties
            ratio = blue_length/red_length if red_length > 0 else 0
            ratio_error = ratio * np.sqrt((L_error/L_km)**2 + (a_error/a_km)**2)
            
            # Twist calculations
            T_TDm = 0.26 * ratio - 0.15
            T_obs = 0.21 * ratio - 0.81
            
            # Propagated twist uncertainties
            T_TDm_error_prop = 0.26 * ratio_error
            T_obs_error_prop = 0.21 * ratio_error
            
            return (L_km, a_km, T_TDm, T_obs, 
                    L_error, a_error, 
                    T_TDm_error_prop, T_obs_error_prop,
                    T_TDm_std, T_obs_std)
            
        except Exception as e:
            print(f"Measurement calculation failed: {e}")
            return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
   
    def store_measurement(self):
        """Store current measurement with uncertainties and prepare for next."""
        if not self.current_lines['blue'] or not self.current_lines['red']:
            print("Please draw both blue (length) and red (width) lines before storing measurement")
            return
            
        results = self.calculate_measurements()
        L_km, a_km, T_TDm, T_obs = results[:4]
        L_error, a_error = results[4:6]
        T_TDm_error_prop, T_obs_error_prop = results[6:8]
        T_TDm_std, T_obs_std = results[8:]
        
        if L_km > 0 and a_km > 0:
            # Store measurement with all uncertainties
            self.measurements.append((L_km, a_km, T_TDm, T_obs, 
                                    L_error, a_error, 
                                    T_TDm_error_prop, T_obs_error_prop))
            self.all_measurements_lines.append(self.current_lines.copy())
            
            # Print results with both uncertainty types
            print(f"\nMeasurement {len(self.measurements)}:")
            print(f"L: {L_km:.2f} ± {L_error:.2f} km")
            print(f"a: {a_km:.2f} ± {a_error:.2f} km")
            print(f"T_TDm: {T_TDm:.2f} ± {T_TDm_error_prop:.2f} (prop) ± {T_TDm_std:.2f} (std)")
            print(f"T_obs: {T_obs:.2f} ± {T_obs_error_prop:.2f} (prop) ± {T_obs_std:.2f} (std)")
            
            self.save_measurement_image(len(self.measurements))
            self.reset_measurement()

    def save_coordinates(self):
        """Save measurement coordinates to file with heliographic positions."""
        coords_file = os.path.join(self.output_dir, f'coordinates_{self.observatory}_{self.wavelength}.txt')
        
        roi_x1 = self.roi_coords['x']
        roi_y1 = self.roi_coords['y']
        roi_x2 = roi_x1 + self.roi_coords['width']
        roi_y2 = roi_y1 + self.roi_coords['height']
        
        with open(coords_file, 'w') as f:
            f.write(f"# Measurement Coordinates File\n")
            f.write(f"# Observatory: {self.observatory}\n")
            f.write(f"# Wavelength: {self.wavelength}\n")
            f.write(f"# Observation Time: {self.timestamp}\n")
            f.write(f"# ROI: (x1={roi_x1}, y1={roi_y1}, x2={roi_x2}, y2={roi_y2})\n\n")
            f.write("Measurement,Line_Type,X,Y,Lat,Lon\n")
            
            for m_idx, measurement in enumerate(self.all_measurements_lines, 1):
                for color in ['blue', 'red']:
                    for line in measurement[color]:
                        x1, y1 = line[0][0] + roi_x1, line[0][1] + roi_y1
                        x2, y2 = line[1][0] + roi_x1, line[1][1] + roi_y1
                        
                        # Account for downsampling factor
                        orig_x1 = int(x1 * self.downsample_factor)
                        orig_y1 = int(y1 * self.downsample_factor)
                        orig_x2 = int(x2 * self.downsample_factor)
                        orig_y2 = int(y2 * self.downsample_factor)
                        
                        # Use assume_spherical_screen context manager
                        from sunpy.coordinates.frames import Helioprojective
                        with Helioprojective.assume_spherical_screen(self.euv_map.observer_coordinate):
                            # Convert to world coordinates
                            coord1 = self.euv_map.pixel_to_world(orig_x1 * u.pixel, 
                                (self.euv_map.data.shape[0] - orig_y1) * u.pixel)
                            coord2 = self.euv_map.pixel_to_world(orig_x2 * u.pixel, 
                                (self.euv_map.data.shape[0] - orig_y2) * u.pixel)
                            
                            stonyhurst1 = coord1.transform_to('heliographic_stonyhurst')
                            stonyhurst2 = coord2.transform_to('heliographic_stonyhurst')
                        
                        f.write(f"{m_idx},{color},{orig_x1},{orig_y1},{stonyhurst1.lat.value:.2f},{stonyhurst1.lon.value:.2f}\n")
                        f.write(f"{m_idx},{color},{orig_x2},{orig_y2},{stonyhurst2.lat.value:.2f},{stonyhurst2.lon.value:.2f}\n")

            


    def save_measurement_image(self, measurement_num):
        """Save a high-resolution version using the full FITS data with heliographic grid."""
        try:
            # Get timestamps for both images
            euv_time = self.euv_map.date.strftime('%Y-%m-%dT%H:%M:%S')
            mag_time = self.mag_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Import for the non-linear stretching
            from astropy.visualization import ImageNormalize, AsinhStretch
            
            # Use the EXACT SAME coordinate handling as the working high-res function
            orig_x = self.orig_roi['x'] if hasattr(self, 'orig_roi') else 0
            orig_y = self.orig_roi['y'] if hasattr(self, 'orig_roi') else 0
            orig_w = self.orig_roi['width'] if hasattr(self, 'orig_roi') else self.euv_map.data.shape[1]
            orig_h = self.orig_roi['height'] if hasattr(self, 'orig_roi') else self.euv_map.data.shape[0]
            
            # CRITICAL FIX: Flip the y-coordinate to account for opposite coordinate systems
            original_height = self.euv_map.data.shape[0]
            corrected_y = original_height - orig_y - orig_h
            
            # Ensure coordinates are within bounds (this handles any edge cases)
            corrected_y = max(0, min(corrected_y, original_height - 1))
            orig_x = max(0, min(orig_x, self.euv_map.data.shape[1] - 1))
            
            # Ensure height and width don't exceed image bounds
            orig_h = min(orig_h, original_height - corrected_y)
            orig_w = min(orig_w, self.euv_map.data.shape[1] - orig_x)
            
            print(f"Extraction - Original ROI: x={orig_x}, y={orig_y}, w={orig_w}, h={orig_h}")
            print(f"Extraction - Corrected ROI: x={orig_x}, y={corrected_y}, w={orig_w}, h={orig_h}")
            
            # Extract data with corrected coordinates
            euv_roi_data = self.euv_map.data[corrected_y:corrected_y+orig_h, orig_x:orig_x+orig_w].copy()
            mag_roi_data = self.mag_map.data[corrected_y:corrected_y+orig_h, orig_x:orig_x+orig_w].copy()
            
            print(f"ROI data shape: {euv_roi_data.shape}")
            
            # Create a header-only submap to get WCS info
            bottom_left = (orig_x, corrected_y) * u.pixel
            top_right = (orig_x + orig_w, corrected_y + orig_h) * u.pixel
            euv_submap = self.euv_map.submap(bottom_left, top_right=top_right)
            mag_submap = self.mag_map.submap(bottom_left, top_right=top_right)
            
            # Calculate aspect ratio
            aspect_ratio = orig_h / orig_w
            
            # Create figure for plotting with higher DPI for better resolution
            # Use aspect ratio to determine figure height
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*aspect_ratio, 5*aspect_ratio),gridspec_kw={'wspace': 0.01})
            
            # OPTION 2: Use AsinhStretch for the EUV data
            # This provides better balance between dark and bright regions
            
            # Get reasonable data limits for normalization
            vmin = np.percentile(euv_roi_data[np.isfinite(euv_roi_data)], 1)
            vmax = np.percentile(euv_roi_data[np.isfinite(euv_roi_data)], 99)
            
            # Create a normalization with arcsinh stretch
            # The parameter 0.1 controls the strength - adjust between 0.05 and 0.5 for different effects
            norm = ImageNormalize(euv_roi_data, stretch=AsinhStretch(0.05), vmin=vmin, vmax=vmax)
            
            # Plot with the non-linear normalization
            im1 = ax1.imshow(euv_roi_data, origin='lower', cmap='sdoaia171', norm=norm)
            
            # Plot the magnetogram data (unchanged)
            im2 = ax2.imshow(mag_roi_data, origin='lower', cmap='RdBu_r', vmin=-500, vmax=500)
            
            # Set axis limits to match image dimensions exactly
            ax1.set_xlim(-0.5, orig_w-0.5)
            ax1.set_ylim(-0.5, orig_h-0.5)
            ax2.set_xlim(-0.5, orig_w-0.5)
            ax2.set_ylim(-0.5, orig_h-0.5)
            
            # Set up custom lon/lat grid
            fontsize_labels = 20
            fontsize_ticks = 20
            title_fontsize = 20
            
            # Add formal solar coordinate labels
            #ax1.set_xlabel('Helioprojective Longitude (Solar-X)', fontsize=fontsize_labels)
            ax1.set_ylabel('Helioprojective Latitude (Solar-Y)', fontsize=fontsize_labels)
            #ax2.set_xlabel('Helioprojective Longitude (Solar-X)', fontsize=fontsize_labels)
            
            # Remove y-axis from magnetogram
            ax2.set_ylabel('')
            ax2.tick_params(left=False, labelleft=False)
            ax2.set_yticks([])
            
            # Set tick sizes
            ax1.tick_params(axis='both', labelsize=fontsize_ticks)
            ax2.tick_params(axis='x', labelsize=fontsize_ticks)
            
            # Create evenly spaced ticks that don't overlap
            total_ticks = 5
            x_ticks = np.linspace(0, orig_w-1, total_ticks)
            y_ticks = np.linspace(0, orig_h-1, total_ticks)
            
            # Get the pixel scale from the map
            pixel_scale_x = self.euv_map.scale[0].value  # arcsec/pixel
            pixel_scale_y = self.euv_map.scale[1].value  # arcsec/pixel
            
            # Calculate arcsecond values at each tick - round to integers
            x_arcsec = np.round(x_ticks * pixel_scale_x).astype(int)
            y_arcsec = np.round(y_ticks * pixel_scale_y).astype(int)
            
            # Set arcsecond ticks avoiding overlap between plots
            ax1.set_xticks(x_ticks[:-1])  # Skip last tick on left plot
            ax1.set_yticks(y_ticks)
            ax1.set_xticklabels([f"{x}\"" for x in x_arcsec[:-1]])  # Add " symbol to each label
            ax1.set_yticklabels([f"{y}\"" for y in y_arcsec])  # Add " symbol to each label
                        
            ax2.set_xticks(x_ticks)  # Skip first tick on right plot
            ax2.set_xticklabels([f"{x}\"" for x in x_arcsec])  # Add " symbol to each label
            
            # Add titles
            ax1.set_title(f'{self.observatory}/{self.wavelength}\n{euv_time}', fontsize=title_fontsize)
            ax2.set_title(f'SDO/HMI\n{mag_time}', fontsize=title_fontsize)
            # Create a shared x-axis label centered at the bottom of the figure
            # Adjust the y position (0.01) as needed to position it correctly
            fig.text(0.5, 0.01, 'Helioprojective Longitude (Solar-X)', 
                     ha='center', va='bottom', fontsize=fontsize_labels)
            
            # Add latitude/longitude grid
            try:
                # Import SkyCoord from astropy (not sunpy)
                from astropy.coordinates import SkyCoord
                
                # Define grid spacing in degrees
                lon_spacing = 10  # degrees
                lat_spacing = 10  # degrees
                
                # Get the heliographic bounding box of our ROI
                bottom_left_world = euv_submap.pixel_to_world(0*u.pixel, 0*u.pixel)
                top_right_world = euv_submap.pixel_to_world((orig_w-1)*u.pixel, (orig_h-1)*u.pixel)
                
                # Convert to heliographic coordinates
                bottom_left_hg = bottom_left_world.transform_to('heliographic_stonyhurst')
                top_right_hg = top_right_world.transform_to('heliographic_stonyhurst')
                
                # Get the latitude and longitude ranges
                lon_min = min(bottom_left_hg.lon.value, top_right_hg.lon.value)
                lon_max = max(bottom_left_hg.lon.value, top_right_hg.lon.value)
                lat_min = min(bottom_left_hg.lat.value, top_right_hg.lat.value)
                lat_max = max(bottom_left_hg.lat.value, top_right_hg.lat.value)
                
                # Round to nearest grid lines
                lon_min = np.floor(lon_min / lon_spacing) * lon_spacing
                lon_max = np.ceil(lon_max / lon_spacing) * lon_spacing
                lat_min = np.floor(lat_min / lat_spacing) * lat_spacing
                lat_max = np.ceil(lat_max / lat_spacing) * lat_spacing
                
                # Create latitude and longitude grids
                lons = np.arange(lon_min, lon_max + lon_spacing, lon_spacing)
                lats = np.arange(lat_min, lat_max + lat_spacing, lat_spacing)
                
                # Create a meshgrid of points for each longitude line
                for lon in lons:
                    # Create points along this longitude line
                    grid_lats = np.linspace(lat_min, lat_max, 100)
                    grid_points = []
                    for lat in grid_lats:
                        # Create a SkyCoord for this point
                        hg_point = SkyCoord(
                            lon * u.deg, 
                            lat * u.deg, 
                            frame='heliographic_stonyhurst', 
                            obstime=euv_submap.date
                        )
                        
                        # Convert to pixel coordinates in our ROI's coordinate system
                        pixel_point = euv_submap.world_to_pixel(hg_point)
                        if (0 <= pixel_point.x.value < orig_w) and (0 <= pixel_point.y.value < orig_h):
                            grid_points.append((pixel_point.x.value, pixel_point.y.value))
                    
                    # Plot the longitude line
                    if grid_points:
                        points = np.array(grid_points)
                        ax1.plot(points[:, 0], points[:, 1], 'gray', alpha=0.5, linestyle='--', linewidth=0.5)
                        ax2.plot(points[:, 0], points[:, 1], 'gray', alpha=0.5, linestyle='--', linewidth=0.5)
                
                # Create a meshgrid of points for each latitude line
                for lat in lats:
                    # Create points along this latitude line
                    grid_lons = np.linspace(lon_min, lon_max, 100)
                    grid_points = []
                    for lon in grid_lons:
                        # Create a SkyCoord for this point
                        hg_point = SkyCoord(
                            lon * u.deg, 
                            lat * u.deg, 
                            frame='heliographic_stonyhurst', 
                            obstime=euv_submap.date
                        )
                        
                        # Convert to pixel coordinates in our ROI's coordinate system
                        pixel_point = euv_submap.world_to_pixel(hg_point)
                        if (0 <= pixel_point.x.value < orig_w) and (0 <= pixel_point.y.value < orig_h):
                            grid_points.append((pixel_point.x.value, pixel_point.y.value))
                    
                    # Plot the latitude line
                    if grid_points:
                        points = np.array(grid_points)
                        ax1.plot(points[:, 0], points[:, 1], 'gray', alpha=0.5, linestyle='--', linewidth=0.5)
                        ax2.plot(points[:, 0], points[:, 1], 'gray', alpha=0.5, linestyle='--', linewidth=0.5)
                
                    # Add grid labels
                    for lon in lons:
                        # Try to find a point at the middle latitude to place the label
                        mid_lat = (lat_min + lat_max) / 2
                        hg_point = SkyCoord(
                            lon * u.deg, 
                            mid_lat * u.deg, 
                            frame='heliographic_stonyhurst', 
                            obstime=euv_submap.date
                        )
                        pixel_point = euv_submap.world_to_pixel(hg_point)
                        if (0 <= pixel_point.x.value < orig_w) and (0 <= pixel_point.y.value < orig_h):
                            ax1.text(pixel_point.x.value, 10, f"{lon:.0f}°", color='gray', 
                                     fontsize=fontsize_ticks-6, ha='center', va='bottom', alpha=0.7)
                            ax2.text(pixel_point.x.value, 10, f"{lon:.0f}°", color='gray', 
                                     fontsize=fontsize_ticks-6, ha='center', va='bottom', alpha=0.7)
                    
                    for lat in lats:
                        # Try to find a point at the middle longitude to place the label
                        mid_lon = (lon_min + lon_max) / 2
                        hg_point = SkyCoord(
                            mid_lon * u.deg, 
                            lat * u.deg, 
                            frame='heliographic_stonyhurst', 
                            obstime=euv_submap.date
                        )
                        pixel_point = euv_submap.world_to_pixel(hg_point)
                        if (0 <= pixel_point.x.value < orig_w) and (0 <= pixel_point.y.value < orig_h):
                            ax1.text(10, pixel_point.y.value, f"{lat:.0f}°", color='gray', 
                                     fontsize=fontsize_ticks-6, ha='left', va='center', alpha=0.7)
                            ax2.text(10, pixel_point.y.value, f"{lat:.0f}°", color='gray', 
                                     fontsize=fontsize_ticks-6, ha='left', va='center', alpha=0.7)
                    
                    print("Successfully added heliographic grid")
                
            except Exception as grid_error:
                print(f"Could not add heliographic grid: {grid_error}")
            
            # Draw measurement lines
            current_measurement = self.all_measurements_lines[-1]
            for color in ['blue', 'red']:
                for line in current_measurement[color]:
                    # IMPORTANT: The line coordinates need to be transformed to match the imshow coordinate system
                    # which has origin='lower' (0,0 at bottom left)
                    x1, y1 = line[0][0], orig_h - line[0][1] - 1  # -1 for zero-indexing
                    x2, y2 = line[1][0], orig_h - line[1][1] - 1
                    
                    # Ensure coordinates are within the plotted region
                    x1 = max(0, min(x1, orig_w - 1))
                    y1 = max(0, min(y1, orig_h - 1))
                    x2 = max(0, min(x2, orig_w - 1))
                    y2 = max(0, min(y2, orig_h - 1))
                    
                    c = 'blue' if color == 'blue' else 'red'
                    linestyle = ':' if color == 'blue' else '-'  # Dotted for blue, solid for red
                    
                    ax1.plot([x1, x2], [y1, y2], color=c, linewidth=2, linestyle=linestyle)
                    ax2.plot([x1, x2], [y1, y2], color=c, linewidth=2, linestyle=linestyle)
            
            # Apply tight_layout to reduce whitespace
            plt.tight_layout(pad=1, w_pad=0.01)
            
            # Save high-resolution image
            output_filename = f'measurement_{measurement_num}_{self.observatory}_{self.wavelength}.png'
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight', dpi=600)  # Higher DPI for better resolution
            print(f"Saved image to: {output_path}")
            
            # Also save as PDF for publication quality
            pdf_filename = f'measurement_{measurement_num}_{self.observatory}_{self.wavelength}.pdf'
            pdf_path = os.path.join(self.output_dir, pdf_filename)
            plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
            print(f"Saved PDF to: {pdf_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Error saving image: {e}")
            import traceback
            traceback.print_exc()  # Print the full traceback for better debugging
            print("Attempting fallback method...")
            self.save_fallback_measurement_image(measurement_num)





    def save_fallback_measurement_image(self, measurement_num):
        """Simple fallback method if the advanced image saving fails"""
        try:
            # Create a simple image without coordinate transformations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Just display the raw images
            ax1.imshow(self.euv_display, origin='upper')
            ax2.imshow(self.mag_display, origin='upper')
            
            # Add simple titles
            ax1.set_title(f'{self.observatory}/{self.wavelength}Å')
            ax2.set_title('HMI Magnetogram')
            
            # Draw the measurement lines directly on display images
            euv_width = self.euv_display.shape[1]
            current_measurement = self.all_measurements_lines[-1]
            
            for color in ['blue', 'red']:
                for line in current_measurement[color]:
                    c = 'blue' if color == 'blue' else 'red'
                    linestyle = '--' if color == 'blue' else '-'
                    ax1.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                             color=c, linewidth=3, linestyle=linestyle)
                    ax2.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 
                             color=c, linewidth=3, linestyle=linestyle)
            
            plt.tight_layout()
            
            # Save fallback image
            output_filename = f'fallback_measurement_{measurement_num}_{self.observatory}.png'
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, bbox_inches='tight')
            print(f"Saved fallback image to: {output_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"Matplotlib fallback method failed: {e}")
            self.save_emergency_fallback_image(measurement_num)
    
    def save_emergency_fallback_image(self, measurement_num):
        """Emergency fallback method using OpenCV directly if matplotlib fails"""
        try:
            print("Using emergency fallback with OpenCV...")
            # Create a simple OpenCV image with measurement lines
            euv_time = self.euv_map.date.strftime('%Y-%m-%dT%H:%M:%S')
            mag_time = self.mag_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Use a more basic approach without SkyCoord conversions
            display_copy = self.combined_display.copy()
            
            # Add measurement information as text on the image
            measurement = self.measurements[-1]
            L_km, a_km, T_TDm, T_obs = measurement[:4]
            L_err, a_err = measurement[4:6]
            
            info_text = [
                f"EUV: {self.observatory}/{self.wavelength}Å - {euv_time}",
                f"MAG: SDO/HMI - {mag_time}",
                f"L = {L_km:.1f} ± {L_err:.1f} km",
                f"a = {a_km:.1f} ± {a_err:.1f} km",
                f"T_TDm = {T_TDm:.2f}",
                f"T_obs = {T_obs:.2f}"
            ]
            
            # Draw the text at the bottom of the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            y_offset = display_copy.shape[0] - 150
            for i, line in enumerate(info_text):
                cv2.putText(display_copy, line, (20, y_offset + i*30),
                           font, 0.8, (255,255,255), 2)
            
            # Save the result
            fallback_filename = f'measurement_{measurement_num}_{self.observatory}_{self.wavelength}_emergency.png'
            cv2.imwrite(os.path.join(self.output_dir, fallback_filename), display_copy)
            print(f"Saved emergency fallback image")
            
        except Exception as e2:
            print(f"Even emergency fallback visualization failed: {e2}")
            import traceback
            traceback.print_exc()

    def print_final_results(self):
        """Print final measurement statistics."""
        if self.measurements:
            avg_L = np.mean([m[0] for m in self.measurements])
            avg_a = np.mean([m[1] for m in self.measurements])
            avg_T_TDm = np.mean([m[2] for m in self.measurements])
            avg_T_obs = np.mean([m[3] for m in self.measurements])
            
            std_T_TDm = np.std([m[2] for m in self.measurements])
            std_T_obs = np.std([m[3] for m in self.measurements])
            
            _, _, _, _, _, _, T_TDm_prop, T_obs_prop = self.measurements[-1]
            
            print("\nFinal Results:")
            print(f"Total measurements: {len(self.measurements)}")
            print(f"Average L: {avg_L:.2f} km")
            print(f"Average a: {avg_a:.2f} km")
            print(f"Average T_TDm: {avg_T_TDm:.2f} ± {T_TDm_prop:.2f} (prop) ± {std_T_TDm:.2f} (std)")
            print(f"Average T_obs: {avg_T_obs:.2f} ± {T_obs_prop:.2f} (prop) ± {std_T_obs:.2f} (std)")

    def save_results(self):
        """Save all measurements and uncertainties to file."""
        results_file = os.path.join(self.output_dir, f'results_{self.observatory}_{self.wavelength}.txt')
        
        with open(results_file, 'w') as f:
            # Enhanced header with more metadata
            f.write(f"Twist Measurements Results\n")
            f.write(f"------------------------\n")
            f.write(f"Observatory: {self.observatory}\n")
            f.write(f"Wavelength: {self.wavelength}\n")
            f.write(f"EUV Observation Time: {self.timestamp}\n")
            f.write(f"Magnetogram Time: {self.mag_time}\n")
            
            # Handle time difference calculation properly
            try:
                # For astropy Time objects
                time_diff_sec = (self.timestamp - self.mag_time).sec
                time_diff = time_diff_sec / 60.0
                f.write(f"Time difference: {abs(time_diff):.1f} minutes\n\n")
            except Exception as e:
                # Fallback if time difference calculation fails
                f.write(f"EUV time: {self.timestamp}\n")
                f.write(f"Magnetogram time: {self.mag_time}\n\n")
                print(f"Warning: Could not calculate time difference: {e}")
            f.write(f"Time difference: {abs(time_diff):.1f} minutes\n\n")   
            
            # Individual measurements with both error types
            for i, (L_km, a_km, tdm, obs, L_err, a_err, tdm_err_prop, obs_err_prop) in enumerate(self.measurements, 1):
                # Calculate std dev up to current measurement
                if i > 1:
                    tdm_std = np.std([m[2] for m in self.measurements[:i]])
                    obs_std = np.std([m[3] for m in self.measurements[:i]])
                else:
                    tdm_std = obs_std = 0.0
                    
                f.write(f"\nMeasurement {i}:\n")
                f.write(f"L: {L_km:.2f} ± {L_err:.2f} km\n")
                f.write(f"a: {a_km:.2f} ± {a_err:.2f} km\n")
                f.write(f"Ratio L/a: {L_km/a_km:.2f}\n")
                f.write(f"T_TDm: {tdm:.2f} ± {tdm_err_prop:.2f} (prop) ± {tdm_std:.2f} (std)\n")
                f.write(f"T_obs: {obs:.2f} ± {obs_err_prop:.2f} (prop) ± {obs_std:.2f} (std)\n")
            
            # Final averages with both error types
            if self.measurements:
                avg_L = np.mean([m[0] for m in self.measurements])
                avg_a = np.mean([m[1] for m in self.measurements])
                avg_T_TDm = np.mean([m[2] for m in self.measurements])
                avg_T_obs = np.mean([m[3] for m in self.measurements])
                
                # Standard error of the mean = std / sqrt(n)
                std_T_TDm = np.std([m[2] for m in self.measurements]) / np.sqrt(len(self.measurements))
                std_T_obs = np.std([m[3] for m in self.measurements]) / np.sqrt(len(self.measurements))
                
                _, _, _, _, _, _, T_TDm_prop, T_obs_prop = self.measurements[-1]
                
                f.write("\nFinal Results:\n")
                f.write(f"Total measurements: {len(self.measurements)}\n")
                f.write(f"Average L: {avg_L:.2f} km\n")
                f.write(f"Average a: {avg_a:.2f} km\n")
                f.write(f"Average ratio L/a: {avg_L/avg_a:.2f}\n")
                f.write(f"Average T_TDm: {avg_T_TDm:.2f} ± {T_TDm_prop:.2f} (err prop) ± {std_T_TDm:.2f} (err std)\n")
                f.write(f"Average T_obs: {avg_T_obs:.2f} ± {T_obs_prop:.2f} (err prop) ± {std_T_obs:.2f} (err std)\n")
            
        print(f"Results saved to: {results_file}")

    def show_help(self):
        """Show help overlay on the measurement view."""
        help_img = self.combined_display.copy()
        h, w = help_img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        
        # Add help text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 50
        line_height = 30
        
        help_lines = [
            "TWIST MEASUREMENT TOOL HELP",
            "",
            "KEY CONTROLS:",
            "b: Switch to BLUE line mode (Length L measurement)",
            "r: Switch to RED line mode (Width a measurement)",
            "c: Clear current lines",
            "m: Store measurement and start new",
            "q: Quit and save all results",
            "h: Show/hide this help",
            "",
            "MOUSE CONTROLS:",
            "Click and drag to draw measurement lines",
            "",
            "TRACKBAR:",
            "Adjust line thickness with the slider",
            "",
            "Press any key to close this help"
        ]
        
        for line in help_lines:
            cv2.putText(overlay, line, (30, y_pos), font, 0.7, (255, 255, 255), 1)
            y_pos += line_height
        
        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, help_img, 1 - alpha, 0, help_img)
        
        # Show help overlay
        cv2.imshow('Help', help_img)
        cv2.waitKey(0)
        cv2.destroyWindow('Help')
    
    def run(self):
        """Main execution loop."""
        print("\nTwist Measurement Controls:")
        print("---------------------------")
        print("'b': Blue line mode (Length L measurement)")
        print("'r': Red line mode (Width a measurement)")
        print("'c': Clear current measurement")
        print("'m': Store measurement")
        print("'q': Quit and save results")
        print("'h': Show help")
        
        # Create resizable window for high-resolution display
        cv2.namedWindow('Measurement View', cv2.WINDOW_NORMAL)
        
        # Calculate reasonable initial window size
        # Just use default values since the window is resizable anyway
        screen_width = 1600
        screen_height = 1000
        
        # Set initial window size as a reasonable fraction of screen size
        window_width = min(screen_width - 100, self.combined_display.shape[1])
        window_height = min(screen_height - 100, self.combined_display.shape[0])
        
        cv2.resizeWindow('Measurement View', window_width, window_height)
        print(f"Created resizable window. Full resolution image can be zoomed as needed.")
        
        while True:
            cv2.imshow('Measurement View', self.combined_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('b'):
                self.mode = 'blue'
                print("Mode: Length (L) measurement")
            elif key == ord('r'):
                self.mode = 'red'
                print("Mode: Width (a) measurement")
            elif key == ord('c'):
                self.reset_measurement()
            elif key == ord('m'):
                self.store_measurement()
            elif key == ord('h'):
                self.show_help()
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        self.save_coordinates()  
        self.save_results()     
        self.print_final_results()

def show_splash_screen():
    """Show splash screen while initializing."""
    root = tk.Tk()
    root.withdraw()
    
    splash = tk.Toplevel(root)
    splash.title("EUV Twist Measurement Tool")
    splash.geometry("500x300")
    
    # Center the splash screen
    splash.update_idletasks()
    width = splash.winfo_width()
    height = splash.winfo_height()
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")
    
    # Add content to splash screen
    tk.Label(splash, text="EUV Flux Rope Twist Measurement Tool", 
           font=("Helvetica", 16), pady=20).pack()
    
    tk.Label(splash, text="Initializing...", pady=10).pack()
    
    progress_var = tk.DoubleVar()
    progress = tk.ttk.Progressbar(splash, variable=progress_var, length=400)
    progress.pack(pady=20)
    
    # Get input information and return it
    euv_path = None
    mag_path = None
    output_path = None
    
    def get_input_and_start():
        nonlocal euv_path, mag_path, output_path
        progress_var.set(25)
        
        output_path = simpledialog.askstring("Input", "Enter output directory path:", 
                                          parent=splash)
        if not output_path:
            output_path = os.getcwd()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        progress_var.set(50)
        
        euv_path = simpledialog.askstring("Input", "Enter EUV FITS file path (or leave empty to download):", 
                                        parent=splash)
        
        progress_var.set(75)
        
        mag_path = simpledialog.askstring("Input", "Enter HMI FITS file path (or leave empty to download):", 
                                        parent=splash)
        
        progress_var.set(100)
        
        # Close splash and destroy root
        splash.destroy()
        root.destroy()
    
    # Schedule the input collection after showing splash
    splash.after(500, get_input_and_start)
    
    root.mainloop()
    
    return euv_path, mag_path, output_path

if __name__ == "__main__":
    # Create a simple splash screen while initializing
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    splash = tk.Toplevel(root)
    splash.title("EUV Twist Measurement Tool")
    splash.geometry("500x300")
    
    # Center the splash screen
    splash.update_idletasks()
    width = splash.winfo_width()
    height = splash.winfo_height()
    x = (splash.winfo_screenwidth() // 2) - (width // 2)
    y = (splash.winfo_screenheight() // 2) - (height // 2)
    splash.geometry(f"{width}x{height}+{x}+{y}")
    
    # Add content to splash screen
    tk.Label(splash, text="EUV Flux Rope Twist Measurement Tool", 
           font=("Helvetica", 16), pady=20).pack()
    
    progress_var = tk.DoubleVar()
    progress = tk.ttk.Progressbar(splash, variable=progress_var, length=400)
    progress.pack(pady=20)
    
    # Define the update_progress function
    def update_progress(value):
        progress_var.set(value)
        splash.update()
    
    # Function to get input using file dialogs
    def get_input_and_start():
        update_progress(25)
        
        # Ask for output directory with a folder selection dialog
        output_path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=os.getcwd()
        )
        
        if not output_path:
            output_path = os.getcwd()
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        update_progress(50)
        
        # Ask for EUV FITS file with a file selection dialog
        euv_path = filedialog.askopenfilename(
            title="Select EUV FITS File (or Cancel to Download)",
            initialdir=output_path,
            filetypes=[("FITS Files", "*.fits"), ("All Files", "*.*")]
        )
        
        update_progress(75)
        
        # Ask for HMI FITS file with a file selection dialog
        mag_path = filedialog.askopenfilename(
            title="Select HMI Magnetogram FITS File (or Cancel to Download)",
            initialdir=output_path if not euv_path else os.path.dirname(euv_path),
            filetypes=[("FITS Files", "*.fits"), ("All Files", "*.*")]
        )
        
        update_progress(100)
        
        # Close splash and start application
        splash.destroy()
        root.destroy()
        
        # Start the application
        tool = TwistMeasurementWithMag(
            euv_path if euv_path else None,
            mag_path if mag_path else None,
            output_path
        )
        tool.run()
    
    # Schedule the input collection after showing splash
    splash.after(500, get_input_and_start)
    
    root.mainloop()



