subroutine printHeader
implicit none
!
character*10 :: date,time
character(len=*), parameter :: Header = "(&
        '                                                                                '/&
        '********************************************************************************'/&
        '**                                                                            **'/&
        '**                                   WE^2PD                                   **'/&
        '**                                                                            **'/&
        '**    Maximally localized Wannier function based Density Matrix Dynamics in   **'/&
        '**    in the presence of electron-electron and electron-phonon scatterings.   **'/&
        '**                                                                            **'/&
        '**                         Copyright (c)  Yu Zhang, PhD,                      **'/&
        '**                           Email: zhyhku@gmail.com                          **'/&
        '**                                                                            **'/&
        '**   Theoretical Division, Los Alamos National Laboratory, Los Alamos         **'/&
        '**                                                                            **'/&
        '**   This propgram is used to calculate the hot-carrier generation from       **'/&
        '**   generation plasmon decay and its injection to other materials via        **'/&
        '**   the interface. The electron transport is calculated within the NEGF      **'/&
        '**   formalism. The electronic structure employs tight-binding, DFTB or       **'/&
        '**   ab-initio tight-binding (localized wannier function as basis).           **'/&
        '**                                                                            **'/&
        '**                                                                            **'/&
        '********************************************************************************')"

!
! Print Title and Time
!
write(6,Header)
call date_and_time(date,time)
write(6,1000) date(7:8),date(5:6),date(1:4),time(1:2),time(3:4),time(5:6)
!
1000 format( ' Calculation Started on ',' ',A2,"/",A2,"/",A4,"  ",A2,":",A2,":",A2,/)
end subroutine printHeader
!
subroutine printEnding
implicit none
!
character*10 :: date,time
character(len=*), parameter :: Header = "(&
        '********************************************************************************'/&
        '**                                                                            **'/&
        '**                             Program Ended                                  **'/&
        '**                                                                            **'/&
        '********************************************************************************')"
!
call date_and_time(date,time)
write(6,1000) date(7:8),date(5:6),date(1:4),time(1:2),time(3:4),time(5:6)
write(6,Header)
!
1000 format( /,' Calculation Finished on ',' ',A2,"/",A2,"/",A4,"  ",A2,":",A2,":",A2)
end subroutine printEnding
