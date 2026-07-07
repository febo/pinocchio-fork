//! Defines the inline program entrypoint and associated types.

#[cfg(feature = "account-resize")]
use solana_account_view::RuntimeAccount;
use {
    crate::BPF_ALIGN_OF_U128,
    core::{
        ptr::with_exposed_provenance_mut,
        slice::{from_raw_parts, from_raw_parts_mut},
    },
    solana_account_view::AccountView,
    solana_address::Address,
    solana_program_error::ProgramError,
};

/// Maximum number of accounts that can be parsed inline in the entrypoint.
///
/// This is a hard limit to constrain the size of the parsing logic
/// inlined in the entrypoint.
pub const MAX_INLINE_ACCOUNTS: usize = 10;

/// Declare the inline program entrypoint.
///
/// This entrypoint is defined as *inline* because it inlines the account
/// parsing logic into the entrypoint itself. It takes advantage of the fact
/// that the runtime passes the instruction data pointer to the entrypoint.
/// This allows the entrypoint to read the instruction data and decide how many
/// accounts it wants to parse from the program input.
///
/// It offers two macros to declare the entrypoint:
/// [`crate::inline_program_entrypoint!`] and [`crate::execute!`]. The former is
/// used to declare the entrypoint and the latter is used to execute the program
/// logic with a specified number of accounts and a processor function. The
/// processor function is called with the program id, the accounts, and the
/// instruction data.
///
/// The [`crate::inline_program_entrypoint!`] macro is used to declare the
/// entrypoint. The only argument is the name of a function with this type
/// signature:
///
/// ```ignore
/// fn process_instruction(input: ProgramInput) -> ProgramResult;
/// ```
///
/// [`ProgramInput`] offers a method to read the instruction data. Programs can
/// then use the [`crate::execute!`] macro to execute the program logic for the
/// corresponding instruction. The [`crate::execute!`] macro takes three
/// arguments: the number of accounts to parse from the input buffer, the
/// program input, and the name of the processor function. The processor
/// function has this type signature:
///
/// ```ignore
/// fn processor(
///     program_id: &Address,
///     accounts: &mut [AccountView],
///     instruction_data: &[u8]
///  ) -> ProgramResult;
/// ```
///
/// # Example
///
/// Defining an entrypoint and making it conditional on the `bpf-entrypoint`
/// feature. Although the `entrypoint` module is written inline in this example,
/// it is common to put it into its own file.
///
/// ```no_run
/// #[cfg(feature = "bpf-entrypoint")]
/// pub mod entrypoint {
///     use {
///         pinocchio::{
///             entrypoint::inline::ProgramInput, error::ProgramError, execute,
///             inline_program_entrypoint, AccountView, Address, ProgramResult,
///         },
///         pinocchio_system::instructions::CreateAccount,
///     };
///
///     // Declares the entrypoint of the program.
///     inline_program_entrypoint!(process_instruction);
///
///     pub fn process_instruction(input: ProgramInput) -> ProgramResult {
///         match input.data.first() {
///            Some(&0) => execute!((3, input) => create),
///            _ => return Err(ProgramError::InvalidInstructionData),
///        }
///     }
///
///     /// Instruction processor.
///     pub fn create(
///         program_id: &Address,
///         accounts: &mut [AccountView],
///         _instruction_data: &[u8],
///     ) -> ProgramResult {
///         let [from, to, _system_program] = accounts else {
///             return Err(ProgramError::NotEnoughAccountKeys);
///         };
///
///         CreateAccount {
///             from,
///             to,
///             lamports: 1_000_000_000,
///             space: 10,
///             owner: program_id,
///         }
///         .invoke()
///     }
/// }
/// ```
#[macro_export]
macro_rules! inline_program_entrypoint {
    ( $process_instruction:expr ) => {
        /// Program entrypoint.
        #[no_mangle]
        pub unsafe extern "C" fn entrypoint(
            program_input: *mut u8,
            instruction_data: *const u8,
        ) -> u64 {
            match $process_instruction($crate::entrypoint::inline::ProgramInput::new_unchecked(
                program_input,
                instruction_data,
            )) {
                Ok(_) => $crate::SUCCESS,
                Err(error) => error.into(),
            }
        }
    };
}

/// Align a pointer to the BPF alignment of [`u128`].
macro_rules! align_pointer {
    ( $ptr:ident ) => {
        // Integer-to-pointer cast: first compute the aligned address as a `usize`,
        // since this is more CU-efficient than using `ptr::align_offset()` or the
        // strict provenance API (e.g., `ptr::with_addr()`). Then cast the result
        // back to a pointer. The resulting pointer is guaranteed to be valid
        // because it follows the layout serialized by the runtime.
        with_exposed_provenance_mut(
            ($ptr.expose_provenance() + (BPF_ALIGN_OF_U128 - 1)) & !(BPF_ALIGN_OF_U128 - 1),
        )
    };
}

#[cfg(feature = "account-resize")]
macro_rules! store_original_data_len {
    ( $current:expr ) => {{
        let account = *$current;
        let data_len = core::ptr::addr_of!((*account).data_len).read() as u32;

        core::ptr::addr_of_mut!((*account).padding)
            .cast::<u32>()
            .write(data_len);
    }};
}

#[cfg(feature = "account-resize")]
macro_rules! store_original_data_len_and_advance {
    ( $current:ident ) => {{
        store_original_data_len!($current);
        $current = $current.add(1);
    }};
}

/// Convert a `ProgramError` into a `u64`.
///
/// This function is marked as `#[cold]` to move the error conversion from the
/// "hot path" of the entrypoint.
#[cold]
#[inline(never)]
fn not_enough_account_keys() -> ProgramError {
    ProgramError::NotEnoughAccountKeys
}

/// Representation of the program input parameters passed by the runtime to the
/// entrypoint.
#[derive(Debug)]
pub struct ProgramInput {
    /// The data for the instruction.
    pub instruction_data: &'static [u8],

    /// Pointer to the list of account in the input buffer.
    ///
    /// The length is determined by [`Self::available`].
    accounts: *mut AccountView,

    /// The number of accounts available in the input buffer.
    pub available: usize,
}

impl ProgramInput {
    /// Creates a new [`ProgramInput`] for the input buffer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that both `program_input` and `instruction_data`
    /// are valid pointers to the correct locations in the input buffer as
    /// serialized by the SVM loader. The `program_input` pointer should point
    /// to the start of the input buffer, and the `instruction_data` pointer
    /// should point to the start of the instruction data within that buffer.
    #[inline(always)]
    pub unsafe fn new_unchecked(program_input: *mut u8, instruction_data: *const u8) -> Self {
        // Loads the instruction data length (8-bytes before the instruction data).
        let ix_data_len = *(instruction_data.sub(size_of::<u64>()) as *const u64) as usize;
        // The slice of account pointers is located right after the `program_id` +
        // alignment padding. The length of the slice is determined by the first
        // 8-bytes of `program_input`.
        let slice_ptr = instruction_data.add(ix_data_len + size_of::<Address>());

        Self {
            instruction_data: unsafe { from_raw_parts(instruction_data, ix_data_len) },
            accounts: align_pointer!(slice_ptr),
            available: *(program_input as *const u64) as usize,
        }
    }

    /// Return the address of the program.
    #[inline(always)]
    pub fn program_id(&self) -> &Address {
        // SAFETY: The program id is located at the end of the program input buffer
        // serialized by the SVM loader after the instruction data.
        unsafe {
            &*self
                .instruction_data
                .as_ptr()
                .add(self.instruction_data.len())
                .cast::<Address>()
        }
    }

    #[inline(always)]
    pub fn accounts<const N: usize>(&mut self) -> Result<&mut [AccountView], ProgramError> {
        const {
            assert!(
                N <= MAX_INLINE_ACCOUNTS,
                "The maximum number of accounts that can be parsed inline in the entrypoint is \
                 `MAX_INLINE_ACCOUNTS`"
            );
        }

        if N > self.available {
            return Err(not_enough_account_keys());
        }

        #[cfg(feature = "account-resize")]
        {
            let mut current = self.accounts.cast::<*mut RuntimeAccount>();

            match N {
                3 => {
                    unsafe {
                        store_original_data_len_and_advance!(current);
                        store_original_data_len_and_advance!(current);
                        store_original_data_len!(current);
                    };
                }
                2 => {
                    unsafe {
                        store_original_data_len_and_advance!(current);
                        store_original_data_len!(current);
                    };
                }
                1 => {
                    unsafe { store_original_data_len!(current) };
                }
                0 => {
                    // No accounts to process.
                }
                _ => {
                    // `N` is validated against `input.available` and the maximum number
                    // of accounts that can be parsed inline in the entrypoint, which
                    // makes this branch unreachable.
                    unreachable!()
                }
            }
        }

        Ok(unsafe { from_raw_parts_mut(self.accounts, N) })
    }
}

/*
#[inline(always)]
pub fn execute<const N: usize, F>(input: &ProgramInput, processor: F) -> ProgramResult
where
    F: FnOnce(&Address, &mut [AccountView], &[u8]) -> ProgramResult,
{
    const {
        assert!(
            N <= MAX_INLINE_ACCOUNTS,
            "The maximum number of accounts that can be parsed inline in the entrypoint is `MAX_INLINE_ACCOUNTS`"
        );
    }

    if N > input.available {
        return Err(not_enough_account_keys());
    }

    #[cfg(feature = "account-resize")]
    {
        let mut current = input.accounts.cast::<*mut RuntimeAccount>();

        match N {
            3 => {
                unsafe {
                    store_original_data_len_and_advance!(current);
                    store_original_data_len_and_advance!(current);
                    store_original_data_len!(current);
                };
            }
            2 => {
                unsafe {
                    store_original_data_len_and_advance!(current);
                    store_original_data_len!(current);
                };
            }
            1 => {
                unsafe { store_original_data_len!(current) };
            }
            0 => {
                // No accounts to process.
            }
            _ => {
                // `N` is validated against `input.available` and the maximum number
                // of accounts that can be parsed inline in the entrypoint, which
                // makes this branch unreachable.
                unreachable!()
            }
        }
    }

    processor(
        input.program_id(),
        unsafe { from_raw_parts_mut(input.accounts as *mut AccountView, N) },
        input.instruction_data,
    )
}
*/

/*
#[macro_export]
macro_rules! execute {
    ( (1, $context:expr) => $processor:expr ) => {{
        let mut current = $context.accounts;

        store_original_data_len!(current);

        let accounts = unsafe { ::core::slice::from_raw_parts_mut($context.accounts, 1) };

        $processor($context.program_id(), accounts, $context.data)
    }};

    ( (2, $context:expr) => $processor:expr ) => {
        let mut current = $context.accounts;

        store_original_data_len_and_advance!(current);
        store_original_data_len!(current);

        let accounts = unsafe { ::core::slice::from_raw_parts_mut($context.accounts, 1) };

        $processor($context.program_id(), accounts, $context.data)
    };
}
 */
