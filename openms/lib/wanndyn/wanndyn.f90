module WannDyn_class
!
  use parameters
!
  implicit none

  type, abstract :: densitymatrix
      integer :: n_mo
      real(dp), dimension(:, :), allocatable :: rho
  end type densitymatrix

contains

  !abstract interface ! TBA


  subroutine we2pd()
  implicit none


  end subroutine we2pd

end module WannDyn_class
