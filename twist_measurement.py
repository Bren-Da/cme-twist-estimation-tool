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
from datetime import datetime, timedelta 
import sunpy.map
from sunpy.coordinates import frames
import astropy.units as u
from reproject import reproject_interp
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch
import warnings
from astropy.utils.exceptions import AstropyWarning
from sunpy.net import Fido, attrs as a
import sunpy_soar


# Suppress warnings
warnings.filterwarnings('ignore', category=AstropyWarning)

class TwistMeasurementWithMag:
    """Handles simultaneous EUV and magnetogram analysis for flux rope twist measurements."""
    
    def __init__(self, euv_path=None, mag_path=None, output_path=None):
        """Initialize measurement tool."""
        # Basic initialization
        self.euv_path = euv_path
        self.mag_path = mag_path
        self.base_output_path = output_path or os.getcwd()
        
        # Drawing parameters
        self.line_thickness = 1
        self.drawing = False
        self.mode = 'blue'
        self.measurements = []
        self.current_lines = {'blue': [], 'red': []}
        self.current_points = []
        self.all_measurements_lines = []
        
        # Load FITS first to get observatory info
        if euv_path:
            self.euv_map = sunpy.map.Map(self.euv_path)
            self.observatory = self.euv_map.observatory
            print(f"Debug - Observatory from FITS: {self.observatory}")
            
            # Set instrument params dynamically
            self.instrument_params = {
                self.observatory: {
                    'euv': {'vmin': -20, 'vmax': 1300} if self.observatory == 'SDO' 
                           else {'vmin': -20, 'vmax': 3000} if self.observatory == 'STEREO A'
                           else {'vmin': -20, 'vmax':2000},
                    'mag': {'vmax': 500}
                }
            }
        # For EUI, 'vmax':600 para mas brillo
        # For AIA 1500 normamente 
        
        # Load data or enter download mode
        if euv_path is None or mag_path is None:
            self.download_mode()
        else:
            self.load_data()
        
        # Create window and set up interaction after data is loaded
        cv2.namedWindow('Measurement View')
        cv2.createTrackbar('Line Thickness', 'Measurement View', 1, 20, self.on_thickness_change)
        cv2.setMouseCallback('Measurement View', self.draw_line)


    def download_mode(self):
        """Interactive download interface for EUV and HMI data."""
        if self.euv_path:
            print(f"\nUsing provided EUV file")
            try:
                # Get time range from EUV file
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
                _, self.mag_path = self.download_euv_and_mag(time_range, download_dir=download_dir)
                self.load_data()
                
            except Exception as e:
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
                        
                        self.euv_path, self.mag_path = self.download_euv_and_mag(
                            time_range,
                            instrument=inst,
                            wavelength=wave,
                            download_dir=download_dir
                        )
                        
                        self.load_data()
                        
                    except Exception as e:
                        raise ValueError(f"Download failed: {e}")
                else:
                    raise ValueError("Invalid wavelength selection")
            else:
                raise ValueError("Invalid instrument selection")

    def download_euv_and_mag(self, time_range, instrument='aia', wavelength=171*u.angstrom, download_dir=None):
        """Download EUV and HMI data using Fido
        
        Parameters:
            time_range (list): Start and end times
            instrument (str): 'aia', 'secchi', or 'eui'
            wavelength (astropy.units): Wavelength to download
            download_dir (str): Directory to save files
        """
        print(f"\nSearching data for:")
        print(f"Time range: {time_range}")
        print(f"Instrument: {instrument}")
        print(f"Wavelength: {wavelength}")
                
        try:
            # Handle provided EUV path first
            if self.euv_path:
                print("Using provided EUV file")
                temp_map = sunpy.map.Map(self.euv_path)
                self.observatory = temp_map.observatory
                self.instrument_params = {
                    self.observatory: {
                        'euv': {'vmin': -20, 'vmax': 2000} if self.observatory == 'SDO' 
                               else {'vmin': -10, 'vmax': 11000} if self.observatory == 'STEREO A'
                               else {'vmin': -15, 'vmax': 1800},
                        'mag': {'vmax': 1000}
                    }
                }
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
                        
                    euv_query = Fido.search(
                        a.Time(time_range[0], time_range[1]) &
                        a.Instrument("EUI") &
                        a.Level(2) &
                        a.soar.Product(product)
                    )
                # AIA/SECCHI download
                else:
                    euv_query = Fido.search(
                        a.Time(time_range[0], time_range[1]),
                        a.Instrument(instrument),
                        a.Wavelength(wavelength)
                    )
                    
                print(f"Found {len(euv_query)} EUV results")
                if len(euv_query) > 0:
                    euv_path = Fido.fetch(euv_query[0], path=download_dir)[0]
                    print(f"Downloaded EUV file: {os.path.basename(euv_path)}")
                    
                    # Set observatory info
                    temp_map = sunpy.map.Map(euv_path)
                    self.observatory = temp_map.observatory
                    
                    # Set instrument params
                    self.instrument_params = {
                        self.observatory: {
                            'euv': {'vmin': -20, 'vmax': 2000} if self.observatory == 'SDO' 
                                   else {'vmin': -10, 'vmax': 11000} if self.observatory == 'STEREO A'
                                   else {'vmin': -15, 'vmax': 18000},
                            'mag': {'vmax': 1000}
                        }
                    }
                else:
                    raise ValueError("No EUV data found")
                    
                # Use existing HMI path if provided
                if self.mag_path:
                    return euv_path, self.mag_path
                    
                # Download HMI if no path provided
                mag_query = Fido.search(
                    a.Time(time_range[0], time_range[1]),
                    a.Instrument("hmi"),
                    a.Physobs("LOS_magnetic_field")
                )
                
                print(f"Found {len(mag_query)} magnetogram results")
                if len(mag_query) > 0:
                    mag_path = Fido.fetch(mag_query[0][0], path=download_dir)[0]
                    print(f"Downloaded magnetogram file: {os.path.basename(mag_path)}")
                    return euv_path, mag_path
                else:
                    raise ValueError("No HMI data found")
                    
        except Exception as e:
            print(f"Error during download: {e}")
            print("Query results structure:")
            print(euv_query if 'euv_query' in locals() else "No query results")
            raise ValueError(f"Download failed: {str(e)}")



    def load_data(self):
        """Load and prepare EUV and magnetogram data with ROI selection."""
        try:
            # Load EUV map
            self.euv_map = sunpy.map.Map(self.euv_path)
            self.observatory = self.euv_map.observatory
            self.wavelength = self.euv_map.wavelength
            self.timestamp = self.euv_map.date
            
            # Load magnetogram and store original time
            mag_map_raw = sunpy.map.Map(self.mag_path)
            self.mag_time = mag_map_raw.date  # Store original time
            
            # Reproject magnetogram
            out_shape = self.euv_map.data.shape
            mag_reproj = reproject_interp((mag_map_raw.data, mag_map_raw.wcs),
                                        self.euv_map.wcs, 
                                        shape_out=out_shape)
            
            # Store magnetogram map
            self.mag_map = sunpy.map.Map(mag_reproj[0], self.euv_map.wcs)
            self.mag_data = mag_reproj[0]  # Store reprojected data
            
            # Setup output directory
            self.output_dir = os.path.join(self.base_output_path, 
                                         f'twist_measurements_{self.observatory}')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Prepare displays
            self.euv_display = self.normalize_euv_image(self.euv_map.data)
            self.mag_display = self.normalize_magnetogram(self.mag_data)
            self.combined_display = np.hstack((self.euv_display, self.mag_display))
            
            # Select ROI
            self.select_roi()
            
            print(f"Data loaded successfully")
            print(f"EUV time: {self.timestamp}")
            print(f"HMI time: {self.mag_time}")
            
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def select_roi(self):
        """Interactive region of interest selection."""
        print("Select region of interest and press ENTER")
        
        # Show the ROI selection window
        cv2.namedWindow("Select Region", cv2.WINDOW_NORMAL)
        cv2.imshow("Select Region", self.combined_display)
        cv2.waitKey(1)
        
        roi = cv2.selectROI("Select Region", self.combined_display, False)
        cv2.destroyWindow("Select Region")
        
        x, y, w, h = roi
        self.roi_coords = {'x': x, 'y': y, 'width': w, 'height': h}
        print(f"ROI selected: x={x}, y={y}, width={w}, height={h}")
        
        # Crop both images
        self.euv_display = self.euv_display[y:y+h, x:x+w]
        self.mag_display = self.mag_display[y:y+h, x:x+w]
        
        # Update combined display
        self.combined_display = np.hstack((self.euv_display, self.mag_display))
        self.original_combined = self.combined_display.copy()
        
        # Show preview of cropped region
        cv2.namedWindow("Cropped Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Cropped Preview", self.combined_display)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyWindow("Cropped Preview")


    def normalize_euv_image(self, data):
        """Normalize EUV data for display."""
        try:
            print(f"Debug - Raw observatory name: {self.observatory}")
            print(f"Debug - Available observatories: {list(self.instrument_params.keys())}")
            # Convert 'STEREO A' to 'STEREO'
            #observatory = 'STEREO' if 'STEREO' in raw_observatory else raw_observatory.upper()
            #print(f"Debug - Processed observatory name: {observatory}")  
            #params = self.instrument_params[observatory]['euv']
            
            raw_observatory = self.observatory
            # Match case exactly as in instrument_params
            observatory = raw_observatory if raw_observatory in self.instrument_params else raw_observatory.upper()
            params = self.instrument_params[observatory]['euv']
            
            normalized = np.clip(data, params['vmin'], params['vmax'])
            normalized = ((normalized - params['vmin']) / 
                        (params['vmax'] - params['vmin']) * 255)
            normalized = np.flipud(normalized.astype(np.uint8))
            
            zeros = np.zeros_like(normalized)
            green = (normalized * 0.87).astype(np.uint8)
            return cv2.merge([zeros, green, normalized])
                
        except Exception as e:
            raise ValueError(f"EUV normalization failed: {e}")

    def normalize_magnetogram(self, data):
        """Normalize magnetogram data for display."""
        try:
            vmax = self.instrument_params[self.observatory]['mag']['vmax']
            scaled = np.clip(data, -vmax, vmax)
            normalized = ((scaled + vmax) / (2 * vmax) * 255)
            normalized = np.flipud(normalized.astype(np.uint8))
            return cv2.merge([normalized, normalized, normalized])
            
        except Exception as e:
            raise ValueError(f"Magnetogram normalization failed: {e}")

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
                if len(line) >= 2:  # Add this check
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
        print("Current lines:", self.current_lines)  # Add this debug print
        self.combined_display = self.original_combined.copy()
        self.draw_saved_lines()

    def reset_measurement(self):
        """Clear current measurement."""
        self.combined_display = self.original_combined.copy()
        self.current_lines = {'blue': [], 'red': []}  # Proper empty initialization
        self.current_points = []  # Add this line
        print("Measurement cleared")

    def calculate_measurements(self):
        try:
            # Basic calculations
            rsun_arcsec = self.euv_map.rsun_obs.value
            km_per_arcsec = 695700 / rsun_arcsec
            pixel_scale = self.euv_map.scale[0].value
            pixel_uncertainty = pixel_scale * km_per_arcsec
            
            # Length calculations
            blue_length = sum([np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
                             for p1, p2 in self.current_lines['blue']])
            red_length = sum([np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
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
            raise ValueError(f"Measurement calculation failed: {e}")
   
    def store_measurement(self):
        """Store current measurement with uncertainties and prepare for next."""
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
        coords_file = os.path.join(self.output_dir, f'coordinates_{self.observatory}.txt')
        
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
                        
                        coord1 = self.euv_map.pixel_to_world(x1 * u.pixel, 
                            (self.euv_map.data.shape[0] - y1) * u.pixel)
                        coord2 = self.euv_map.pixel_to_world(x2 * u.pixel, 
                            (self.euv_map.data.shape[0] - y2) * u.pixel)
                        
                        stonyhurst1 = coord1.transform_to('heliographic_stonyhurst')
                        stonyhurst2 = coord2.transform_to('heliographic_stonyhurst')
                        
                        f.write(f"{m_idx},{color},{x1},{y1},{stonyhurst1.lat.value:.2f},{stonyhurst1.lon.value:.2f}\n")
                        f.write(f"{m_idx},{color},{x2},{y2},{stonyhurst2.lat.value:.2f},{stonyhurst2.lon.value:.2f}\n")
    
    def save_measurement_image(self, measurement_num):
        """Save measurement image with metadata"""
        # Get timestamps for both images
        euv_time = self.euv_map.date.strftime('%Y-%m-%dT%H:%M:%S')
        mag_time = self.mag_time.strftime('%Y-%m-%dT%H:%M:%S')
        
        fig = plt.figure(figsize=(12, 6))
        
        # Convert OpenCV BGR to RGB
        rgb_euv = cv2.cvtColor(self.euv_display, cv2.COLOR_BGR2RGB)
        rgb_mag = cv2.cvtColor(self.mag_display, cv2.COLOR_BGR2RGB)
        
        # Create subplots
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Display images
        ax1.imshow(rgb_euv)
        ax2.imshow(rgb_mag)
        
        # Add labels and ticks
        ftsz1 = 20
        ax1.set_xlabel('px ["]', fontsize=ftsz1)
        ax1.set_ylabel('px ["]', fontsize=ftsz1)
        ax2.set_xlabel('px ["]', fontsize=ftsz1)
        ax2.set_ylabel('px ["]', fontsize=ftsz1)
        ax1.tick_params(labelsize=ftsz1)
        ax2.tick_params(labelsize=ftsz1)
        
        # Draw measurement lines
        current_measurement = self.all_measurements_lines[-1]
        for color in ['blue', 'red']:
            for line in current_measurement[color]:
                c = 'blue' if color == 'blue' else 'red'
                ax1.plot([line[0][0], line[1][0]], 
                        [line[0][1], line[1][1]], 
                        c=c, linewidth=2)
                ax2.plot([line[0][0], line[1][0]], 
                        [line[0][1], line[1][1]], 
                        c=c, linewidth=2)
        
        # Set titles with timestamps
        ftsz2 = 20
        ax1.set_title(f'{self.observatory}/{self.wavelength}Å\n{euv_time}', 
                     pad=20, fontsize=ftsz2)
        ax2.set_title(f'SDO/HMI\n{mag_time}', 
                     pad=20, fontsize=ftsz2)
        
        plt.tight_layout()
        
        output_filename = f'measurement_{measurement_num}_{self.observatory}.png'
        plt.savefig(os.path.join(self.output_dir, output_filename), 
                    bbox_inches='tight', dpi=300)
        plt.close()

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
        results_file = os.path.join(self.output_dir, f'results_{self.observatory}.txt')
        
        with open(results_file, 'w') as f:
            # Header remains same
            f.write(f"Twist Measurements Results\n")
            f.write(f"Observatory: {self.observatory}\n")
            f.write(f"Wavelength: {self.wavelength}\n")
            f.write(f"Observation Time: {self.timestamp}\n\n")
            
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
                f.write(f"T_TDm: {tdm:.2f} ± {tdm_err_prop:.2f} (prop) ± {tdm_std:.2f} (std)\n")
                f.write(f"T_obs: {obs:.2f} ± {obs_err_prop:.2f} (prop) ± {obs_std:.2f} (std)\n")
            
            # Final averages with both error types
            if self.measurements:
                avg_L = np.mean([m[0] for m in self.measurements])
                avg_a = np.mean([m[1] for m in self.measurements])
                avg_T_TDm = np.mean([m[2] for m in self.measurements])
                avg_T_obs = np.mean([m[3] for m in self.measurements])
                
                std_T_TDm = np.std([m[2] for m in self.measurements]) / np.sqrt(len(self.measurements))
                std_T_obs = np.std([m[3] for m in self.measurements]) / np.sqrt(len(self.measurements))
                
                _, _, _, _, _, _, T_TDm_prop, T_obs_prop = self.measurements[-1]
                
                f.write("\nFinal Results:\n")
                f.write(f"Total measurements: {len(self.measurements)}\n")
                f.write(f"Average L: {avg_L:.2f} km\n")
                f.write(f"Average a: {avg_a:.2f} km\n")
                f.write(f"Average T_TDm: {avg_T_TDm:.2f} ± {T_TDm_prop:.2f} (err prop) ± {std_T_TDm:.2f} (err std)\n")
                f.write(f"Average T_obs: {avg_T_obs:.2f} ± {T_obs_prop:.2f} (err prop) ± {std_T_obs:.2f} (err std)\n")


    def run(self):
        """Main execution loop."""
        print("\nTwist Measurement Controls:")
        print("---------------------------")
        print("'b': Blue line mode (Length L)")
        print("'r': Red line mode (Width a)")
        print("'c': Clear current measurement")
        print("'m': Store measurement")
        print("'q': Quit and save results")
        
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
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        self.save_coordinates()  
        self.save_results()     
        self.print_final_results()

if __name__ == "__main__":
    output_path = input("Enter output directory path: ").strip()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    euv_path = input("Enter EUV FITS file path (or press Enter to download): ").strip()
    mag_path = input("Enter HMI FITS file path (or press Enter to download): ").strip()
    
    tool = TwistMeasurementWithMag(
        euv_path if euv_path else None,
        mag_path if mag_path else None,
        output_path
    )
    tool.run()


# /Users/brendado/Documents/PhD/TWIST-measurements/events/20231128/PRUEBA-VER3
# /Users/brendado/Documents/PhD/TWIST-measurements/events/20231128/PRUEBA-VER3/downloaded_fits/aia_lev1_171a_2023_11_28t19_13_09_34z_image_lev1.fits
# /Users/brendado/Documents/PhD/TWIST-measurements/events/20231128/PRUEBA-VER3/downloaded_fits/hmi_m_45s_2023_11_28_19_14_15_tai_magnetogram.fits

'''
/Users/brendado/Documents/PhD/TWIST-measurements/events/20140910/aia.lev1.171.609318295.2014-09-10T17:17:11.344Z.image_lev1.fits



    def save_measurement_image(self, measurement_num):
        obs_time = self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Create figure with current measurement
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Convert OpenCV BGR to RGB
        rgb_euv = cv2.cvtColor(self.euv_display, cv2.COLOR_BGR2RGB)
        rgb_mag = cv2.cvtColor(self.mag_display, cv2.COLOR_BGR2RGB)
        
        ax1.imshow(rgb_euv)
        ax2.imshow(rgb_mag)
        
        # Draw current measurement lines
        current_measurement = self.all_measurements_lines[-1]
        for color in ['blue', 'red']:
            for line in current_measurement[color]:
                c = 'blue' if color == 'blue' else 'red'
                ax1.plot([line[0][0], line[1][0]], 
                        [line[0][1], line[1][1]], 
                        c=c, linewidth=2)
                ax2.plot([line[0][0], line[1][0]], 
                        [line[0][1], line[1][1]], 
                        c=c, linewidth=2)
        
        ax1.set_title(f'{self.observatory} {self.wavelength}Å')
        ax2.set_title('HMI Magnetogram')
        plt.suptitle(f'Measurement {measurement_num} - {obs_time}')
        
        output_filename = f'measurement_{measurement_num}_{self.observatory}_{obs_time}.png'
        plt.savefig(os.path.join(self.output_dir, output_filename), 
                   bbox_inches='tight', dpi=300)
        plt.close()



    instrument_params = {
        'sdo': {
            'euv': {'vmin': -20, 'vmax': 2000},  # AIA parameters
            'mag': {'vmax': 1000}  # HMI parameters
        },
        'STEREO A': {
            'euv': {'vmin': -10, 'vmax': 11000},  # EUVI parameters
            'mag': {'vmax': 1000}  # HMI reprojected
        },
        'Solar Orbiter': {
            'euv': {'vmin': -15, 'vmax': 18000},  # EUI parameters
            'mag': {'vmax': 1000}  # HMI reprojected
        }
    } 



    def save_measurement_image(self, measurement_num):
        """Save measurement image with metadata"""
        obs_time = self.timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Create figure with more vertical space for title
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Display images
        rgb_euv = cv2.cvtColor(self.euv_display, cv2.COLOR_BGR2RGB)
        rgb_mag = cv2.cvtColor(self.mag_display, cv2.COLOR_BGR2RGB)
        
        ax1.imshow(rgb_euv)
        ax2.imshow(rgb_mag)
        
        # Add axes labels and ticks
        ax1.set_xlabel('X ["]')
        ax1.set_ylabel('Y ["]')
        ax2.set_xlabel('X ["]')
        ax2.set_ylabel('Y ["]')
        
        # Draw measurement lines
        current_measurement = self.all_measurements_lines[-1]
        for color in ['blue', 'red']:
            for line in current_measurement[color]:
                c = 'blue' if color == 'blue' else 'red'
                ax1.plot([line[0][0], line[1][0]], 
                        [line[0][1], line[1][1]], 
                        c=c, linewidth=2)
                ax2.plot([line[0][0], line[1][0]], 
                        [line[0][1], line[1][1]], 
                        c=c, linewidth=2)
        
        # Adjust titles with more padding
        ax1.set_title(f'{self.observatory} {self.wavelength}Å', pad=15, fontsize=18)
        ax2.set_title('HMI Magnetogram', pad=15, fontsize=18)
        
        # Add main title with more space
        plt.suptitle(f'Measurement {measurement_num} - {obs_time}', y=0.9)
        
        plt.tight_layout()
        
        output_filename = f'measurement_{measurement_num}_{self.observatory}_{obs_time}.png'
        plt.savefig(os.path.join(self.output_dir, output_filename), 
                    bbox_inches='tight', dpi=300)
        plt.close()       



'''
