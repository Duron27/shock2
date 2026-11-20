//! The Vulkan 1.0 instance, window surface and logical device.
//!
//! One queue family that does both graphics and present is required; every
//! desktop GPU (and llvmpipe) has one, and it keeps submission single-queue.

use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
use ash::{Device, Entry, Instance, vk};
use tracing::{error, info, warn};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Validation errors reported since start-up. Tests and the smoke example
/// fail on a non-zero count, so a misuse never hides behind a frame that
/// happens to look right.
static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);

pub fn validation_error_count() -> usize {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

/// Validation runs when `SHOCK2_VK_VALIDATION` is set to anything but `0`.
/// It needs the Khronos layer
/// installed (Fedora: `vulkan-validation-layers`) and is skipped otherwise.
fn validation_requested() -> bool {
    // Opt-in: the layer costs a quarter of the frame rate even in a
    // menu-light scene, and dev builds are the ones played.
    matches!(std::env::var("SHOCK2_VK_VALIDATION").as_deref(), Ok(v) if v != "0")
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    // ash 0.37 loads every VK_KHR_swapchain entry point through
    // vkGetDeviceProcAddr, including one instance-level function; harmless.
    if message.contains("vkGetPhysicalDevicePresentRectanglesKHR") {
        return vk::FALSE;
    }
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        error!(target: "vulkan", "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(target: "vulkan", "{message}");
    }
    vk::FALSE
}

pub struct VkContext {
    pub entry: Entry,
    pub instance: Instance,
    debug: Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device_name: String,
    pub limits: vk::PhysicalDeviceLimits,
    /// Whether `samplerAnisotropy` was enabled on the device.
    pub sampler_anisotropy: bool,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family: u32,
    pub device: Device,
    pub queue: vk::Queue,
    pub swapchain_loader: Swapchain,
}

impl VkContext {
    /// `window_extensions` are the instance extensions the window system
    /// needs (GLFW: `get_required_instance_extensions`); `create_surface`
    /// makes the window's surface on the new instance.
    pub fn new(
        app_name: &str,
        window_extensions: &[String],
        create_surface: impl FnOnce(vk::Instance) -> Result<vk::SurfaceKHR, vk::Result>,
    ) -> Result<VkContext, String> {
        let entry = unsafe { Entry::load() }.map_err(|e| format!("no Vulkan loader: {e}"))?;

        let validation = validation_requested()
            && entry
                .enumerate_instance_layer_properties()
                .unwrap_or_default()
                .iter()
                .any(|layer| unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) } == VALIDATION_LAYER);
        if validation_requested() && !validation {
            warn!("Vulkan validation requested but VK_LAYER_KHRONOS_validation is not installed");
        }

        let app_name = CString::new(app_name).unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&app_name)
            .api_version(vk::API_VERSION_1_0);
        let extension_names: Vec<CString> = window_extensions
            .iter()
            .map(|name| CString::new(name.as_str()).unwrap())
            .chain(validation.then(|| DebugUtils::name().to_owned()))
            .collect();
        let extension_ptrs: Vec<*const i8> = extension_names.iter().map(|n| n.as_ptr()).collect();
        let layer_ptrs: Vec<*const i8> = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(&app_info)
                    .enabled_extension_names(&extension_ptrs)
                    .enabled_layer_names(&layer_ptrs),
                None,
            )
        }
        .map_err(|e| format!("vkCreateInstance: {e}"))?;

        let debug = validation.then(|| {
            let loader = DebugUtils::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None) }
                .expect("vkCreateDebugUtilsMessengerEXT");
            (loader, messenger)
        });

        let surface_loader = Surface::new(&entry, &instance);
        let surface =
            create_surface(instance.handle()).map_err(|e| format!("window surface: {e}"))?;

        let (physical_device, queue_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        info!(
            "Vulkan device: {device_name} (validation {})",
            if validation { "on" } else { "off" }
        );

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)
            .build()];
        let supported = unsafe { instance.get_physical_device_features(physical_device) };
        let features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(supported.sampler_anisotropy == vk::TRUE);
        let device_extensions = [Swapchain::name().as_ptr()];
        let device = unsafe {
            instance.create_device(
                physical_device,
                &vk::DeviceCreateInfo::builder()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&device_extensions)
                    .enabled_features(&features),
                None,
            )
        }
        .map_err(|e| format!("vkCreateDevice: {e}"))?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let swapchain_loader = Swapchain::new(&instance, &device);

        Ok(VkContext {
            entry,
            instance,
            debug,
            surface_loader,
            surface,
            physical_device,
            device_name,
            limits: properties.limits,
            sampler_anisotropy: supported.sampler_anisotropy == vk::TRUE,
            memory_properties,
            queue_family,
            device,
            queue,
            swapchain_loader,
        })
    }

    /// A memory type allowed by `type_bits` with all of `flags`.
    pub fn memory_type(&self, type_bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory_properties.memory_type_count).find(|&i| {
            type_bits & (1 << i) != 0
                && self.memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(flags)
        })
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// The best GPU that can present to `surface`: discrete over integrated over
/// virtual over software. `SHOCK2_VK_DEVICE=<name substring>` picks one
/// explicitly (e.g. `llvmpipe` for a software-rendered run).
fn pick_physical_device(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), String> {
    let wanted = std::env::var("SHOCK2_VK_DEVICE")
        .ok()
        .map(|s| s.to_lowercase());
    let devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| format!("vkEnumeratePhysicalDevices: {e}"))?;
    let mut best: Option<(i32, vk::PhysicalDevice, u32)> = None;
    for device in devices {
        let properties = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .to_lowercase();
        if wanted.as_ref().is_some_and(|w| !name.contains(w.as_str())) {
            continue;
        }
        let has_swapchain = unsafe { instance.enumerate_device_extension_properties(device) }
            .unwrap_or_default()
            .iter()
            .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == Swapchain::name());
        if !has_swapchain {
            continue;
        }
        let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
        let Some(family) = families.iter().enumerate().find_map(|(i, family)| {
            let present = unsafe {
                surface_loader.get_physical_device_surface_support(device, i as u32, surface)
            }
            .unwrap_or(false);
            (family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present).then_some(i as u32)
        }) else {
            continue;
        };
        let score = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        if best.is_none_or(|(s, _, _)| score > s) {
            best = Some((score, device, family));
        }
    }
    best.map(|(_, device, family)| (device, family))
        .ok_or_else(|| match wanted {
            Some(w) => format!("no Vulkan device matching SHOCK2_VK_DEVICE={w} can present"),
            None => "no Vulkan device can present to the window".to_owned(),
        })
}
