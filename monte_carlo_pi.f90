!> Estimate pi with a Monte Carlo method.
!>
!> Random points (x, y) are drawn uniformly from the unit square [0,1)^2.
!> The fraction that falls inside the quarter circle x^2 + y^2 <= 1
!> approaches pi/4, so pi ~= 4 * hits / samples.
!>
!> Each sample is a Bernoulli trial with p = pi/4, so the standard error
!> of the estimate is 4 * sqrt(p * (1 - p) / n).
!>
!> Random numbers come from SplitMix64, a counter-based pseudo-random
!> generator: the k-th value of the stream is mix(seed + k * gamma). Any
!> chunk of samples can therefore jump straight to its place in the
!> stream, so each batch is split into fixed-size chunks shared among
!> OpenMP threads and the result for a given seed is identical no matter
!> how many threads run or how the chunks are scheduled.
!>
!> The generator relies on 64-bit integer arithmetic wrapping modulo
!> 2**64, so build with -fwrapv.
!>
!> Build:  gfortran -std=f2008 -O2 -fopenmp -fwrapv -o monte_carlo_pi_f monte_carlo_pi.f90
!> Run:    OMP_NUM_THREADS=8 ./monte_carlo_pi_f [total_samples [seed]]
!>
!> If no seed is given, one is chosen at random and printed so the run
!> can be repeated.
program monte_carlo_pi
   use, intrinsic :: iso_fortran_env, only: int64, real64, output_unit, error_unit
   !$ use omp_lib, only: omp_get_max_threads
   implicit none

   integer(int64), parameter :: default_samples = 100000000_int64
   integer(int64), parameter :: batch_size = 1000000_int64
   ! Samples per unit of parallel work.
   integer(int64), parameter :: chunk_size = 4096_int64
   real(real64), parameter :: pi_ref = acos(-1.0_real64)

   ! SplitMix64 constants (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,
   ! 0x94D049BB133111EB) written as two's-complement signed int64.
   integer(int64), parameter :: sm_gamma = -7046029254386353131_int64
   integer(int64), parameter :: sm_mul1 = -4658895280553007687_int64
   integer(int64), parameter :: sm_mul2 = -7723592293110705685_int64

   integer(int64) :: total_samples, seed, done, hits, n, next_report, i
   integer :: num_threads
   real(real64) :: pi_est, p, std_err

   total_samples = read_int_arg(1, 'sample count', min_value=1_int64, &
                                fallback=default_samples)
   seed = read_int_arg(2, 'seed', min_value=0_int64, fallback=random_seed_value())

   num_threads = 1
   !$ num_threads = omp_get_max_threads()

   write (output_unit, '(a, i0, a, i0, a)') 'Estimating pi with ', total_samples, &
      ' random samples on ', num_threads, ' thread(s)'
   write (output_unit, '(a, i0)') 'Seed: ', seed
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
      !$omp parallel do schedule(static) reduction(+:hits)
      do i = 0, n - 1, chunk_size
         hits = hits + count_hits(done + i, min(chunk_size, n - i))
      end do
      !$omp end parallel do
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

   !> Return command-line argument number pos as an integer, or the
   !> fallback if it is absent. An invalid argument is an error: stop
   !> rather than silently running a different (possibly much longer) job.
   function read_int_arg(pos, what, min_value, fallback) result(value)
      integer, intent(in) :: pos
      character(len=*), intent(in) :: what
      integer(int64), intent(in) :: min_value, fallback
      integer(int64) :: value
      character(len=64) :: arg
      integer :: arg_len, arg_stat, stat

      value = fallback
      if (command_argument_count() < pos) return

      call get_command_argument(pos, arg, length=arg_len, status=arg_stat)

      ! Accept only a plain string of digits that fits in the buffer;
      ! list-directed reads would accept things like "10,5" or "/".
      stat = 1
      if (arg_stat == 0 .and. arg_len > 0) then
         if (verify(arg(1:arg_len), '0123456789') == 0) then
            read (arg(1:arg_len), '(i20)', iostat=stat) value
         end if
      end if

      if (stat /= 0 .or. value < min_value) then
         write (error_unit, '(5a, i0, a)') 'Error: invalid ', what, " '", trim(arg), &
            "' (expected an integer from ", min_value, ' to 9223372036854775807)'
         stop 1
      end if
   end function read_int_arg

   !> A non-repeatable seed for when the user does not supply one, drawn
   !> from the intrinsic generator after seeding it from a
   !> processor-dependent source (gfortran 7+ uses OS entropy when
   !> random_seed is called with no arguments).
   function random_seed_value() result(s)
      integer(int64) :: s
      real(real64) :: r

      call random_seed()
      call random_number(r)
      s = int(r * 2.0_real64**62, int64)
   end function random_seed_value

   !> SplitMix64 output function: scramble a stream state into 64 random
   !> bits and keep the top 53 as a double in [0, 1).
   elemental function splitmix_unit(state) result(u)
      integer(int64), intent(in) :: state
      real(real64) :: u
      integer(int64) :: z

      z = state
      z = ieor(z, ishft(z, -30)) * sm_mul1
      z = ieor(z, ishft(z, -27)) * sm_mul2
      z = ieor(z, ishft(z, -31))
      u = real(ishft(z, -11), real64) * 2.0_real64**(-53)
   end function splitmix_unit

   !> Draw m points starting at global sample index start and return how
   !> many fall inside the quarter circle. Sample j uses stream values
   !> 2j+1 and 2j+2, so the result depends only on the seed and the
   !> sample indices, never on which thread does the work.
   function count_hits(start, m) result(inside)
      integer(int64), intent(in) :: start, m
      integer(int64) :: inside, k, state
      real(real64) :: x, y

      state = seed + 2_int64 * start * sm_gamma
      inside = 0
      do k = 1, m
         state = state + sm_gamma
         x = splitmix_unit(state)
         state = state + sm_gamma
         y = splitmix_unit(state)
         if (x * x + y * y <= 1.0_real64) inside = inside + 1
      end do
   end function count_hits

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
