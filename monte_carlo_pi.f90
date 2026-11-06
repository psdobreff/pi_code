!> Estimate pi with a Monte Carlo method.
!>
!> Random points (x, y) are drawn uniformly from the unit square [0,1)^2.
!> The fraction that falls inside the quarter circle x^2 + y^2 <= 1
!> approaches pi/4, so pi ~= 4 * hits / samples.
!>
!> Each sample is a Bernoulli trial with p = pi/4, so the standard error
!> of the estimate is 4 * sqrt(p * (1 - p) / n).
!>
!> Build:  gfortran-16 -std=f2023 -O2 -o monte_carlo_pi_f monte_carlo_pi.f90
!> Run:    ./monte_carlo_pi_f [total_samples]
program monte_carlo_pi
   use, intrinsic :: iso_fortran_env, only: int64, real64, output_unit, error_unit
   implicit none

   integer(int64), parameter :: default_samples = 100000000_int64
   integer(int64), parameter :: batch_size = 1000000_int64
   real(real64), parameter :: pi_ref = acos(-1.0_real64)

   integer(int64) :: total_samples, done, hits, n, next_report
   real(real64), allocatable :: x(:), y(:)
   real(real64) :: pi_est, p, std_err

   total_samples = read_sample_count(default_samples)

   ! Seed the intrinsic pseudo-random number generator from a
   ! processor-dependent source (Fortran 2018+).
   call random_init(repeatable=.false., image_distinct=.true.)

   allocate (x(batch_size), y(batch_size))

   write (output_unit, '(a, i0, a)') 'Estimating pi with ', total_samples, ' random samples'
   write (output_unit, '(a)') ''
   write (output_unit, '(a15, 2x, a14, 2x, a12, 2x, a12)') &
      'samples', 'pi estimate', 'abs error', 'std error'

   done = 0
   hits = 0
   next_report = 1000

   do while (done < total_samples)
      ! Stop each batch at the next report point so every power of 10
      ! gets its own row, even below batch_size.
      n = min(batch_size, total_samples - done, next_report - done)
      call random_number(x(1:n))
      call random_number(y(1:n))
      hits = hits + count(x(1:n)**2 + y(1:n)**2 <= 1.0_real64, kind=int64)
      done = done + n

      ! Report at each power of 10 (and at the end) to show convergence.
      if (done == next_report .or. done == total_samples) then
         call report(done, hits)
         if (done == next_report) next_report = next_report * 10
      end if
   end do

   p = real(hits, real64) / real(done, real64)
   pi_est = 4.0_real64 * p
   std_err = 4.0_real64 * sqrt(p * (1.0_real64 - p) / real(done, real64))

   write (output_unit, '(a)') ''
   write (output_unit, '(a, f16.12)') 'Final estimate : ', pi_est
   write (output_unit, '(a, f16.12)') 'Reference pi   : ', pi_ref
   write (output_unit, '(a, es16.6)') 'Absolute error : ', abs(pi_est - pi_ref)
   write (output_unit, '(a, f16.12, a, f16.12, a)') '95% interval   : [', &
      pi_est - 1.96_real64 * std_err, ', ', pi_est + 1.96_real64 * std_err, ']'

contains

   !> Return the sample count from the first command-line argument, or the
   !> fallback if none is given. An invalid argument is an error: stop
   !> rather than silently running a different (possibly much longer) job.
   function read_sample_count(fallback) result(count_out)
      integer(int64), intent(in) :: fallback
      integer(int64) :: count_out
      character(len=64) :: arg
      integer :: arg_len, arg_stat, stat

      count_out = fallback
      if (command_argument_count() < 1) return

      call get_command_argument(1, arg, length=arg_len, status=arg_stat)

      ! Accept only a plain string of digits that fits in the buffer;
      ! list-directed reads would accept things like "10,5" or "/".
      stat = 1
      if (arg_stat == 0 .and. arg_len > 0) then
         if (verify(arg(1:arg_len), '0123456789') == 0) then
            read (arg(1:arg_len), '(i20)', iostat=stat) count_out
         end if
      end if

      if (stat /= 0 .or. count_out < 1) then
         write (error_unit, '(3a)') "Error: invalid sample count '", trim(arg), &
            "' (expected a positive integer, e.g. 1000000)"
         stop 1, quiet=.true.
      end if
   end function read_sample_count

   subroutine report(samples, inside)
      integer(int64), intent(in) :: samples, inside
      real(real64) :: frac, est, se

      frac = real(inside, real64) / real(samples, real64)
      est = 4.0_real64 * frac
      se = 4.0_real64 * sqrt(frac * (1.0_real64 - frac) / real(samples, real64))
      write (output_unit, '(i15, 2x, f14.10, 2x, es12.4, 2x, es12.4)') &
         samples, est, abs(est - pi_ref), se
   end subroutine report

end program monte_carlo_pi
