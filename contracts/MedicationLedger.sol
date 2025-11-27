// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MedicationLedger {
    event PrescriptionLogged(
        string prescriptionId,
        string patientId,
        string medicationId,
        uint256 startDate,
        uint256 endDate,
        bytes32 dataHash
    );

    event IntakeEventLogged(
        string intakeEventId,
        string sessionId,
        string patientId,
        string prescriptionId,
        uint256 intakeTime,
        uint16 confidence1000,
        bytes32 dataHash
    );

    function logPrescription(
        string calldata prescriptionId,
        string calldata patientId,
        string calldata medicationId,
        uint256 startDate,
        uint256 endDate,
        bytes32 dataHash
    ) external {
        emit PrescriptionLogged(
            prescriptionId,
            patientId,
            medicationId,
            startDate,
            endDate,
            dataHash
        );
    }

    function logIntakeEvent(
        string calldata intakeEventId,
        string calldata sessionId,
        string calldata patientId,
        string calldata prescriptionId,
        uint256 intakeTime,
        uint16 confidence1000,
        bytes32 dataHash
    ) external {
        emit IntakeEventLogged(
            intakeEventId,
            sessionId,
            patientId,
            prescriptionId,
            intakeTime,
            confidence1000,
            dataHash
        );
    }
}
