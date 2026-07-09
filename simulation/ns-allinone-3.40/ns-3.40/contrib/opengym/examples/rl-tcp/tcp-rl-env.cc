/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Technische Universität Berlin
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz@tkn.tu-berlin.de>
 */

#include "tcp-rl-env.h"

#include "ns3/core-module.h"
#include "ns3/log.h"
#include "ns3/object.h"
#include "ns3/simulator.h"
#include "ns3/tcp-header.h"
#include "ns3/tcp-socket-base.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ns3::TcpGymEnv");
NS_OBJECT_ENSURE_REGISTERED(TcpGymEnv);

TcpGymEnv::TcpGymEnv()
{
    NS_LOG_FUNCTION(this);
    SetOpenGymInterface(OpenGymInterface::Get());
}

TcpGymEnv::~TcpGymEnv()
{
    NS_LOG_FUNCTION(this);
}

TypeId
TcpGymEnv::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::TcpGymEnv").SetParent<OpenGymEnv>().SetGroupName("OpenGym");

    return tid;
}

void
TcpGymEnv::DoDispose()
{
    NS_LOG_FUNCTION(this);
}

void
TcpGymEnv::SetNodeId(uint32_t id)
{
    NS_LOG_FUNCTION(this);
    m_nodeId = id;
}

void
TcpGymEnv::SetSocketUuid(uint32_t id)
{
    NS_LOG_FUNCTION(this);
    m_socketUuid = id;
}

std::string
TcpGymEnv::GetTcpCongStateName(const TcpSocketState::TcpCongState_t state)
{
    std::string stateName = "UNKNOWN";
    switch (state)
    {
    case TcpSocketState::CA_OPEN:
        stateName = "CA_OPEN";
        break;
    case TcpSocketState::CA_DISORDER:
        stateName = "CA_DISORDER";
        break;
    case TcpSocketState::CA_CWR:
        stateName = "CA_CWR";
        break;
    case TcpSocketState::CA_RECOVERY:
        stateName = "CA_RECOVERY";
        break;
    case TcpSocketState::CA_LOSS:
        stateName = "CA_LOSS";
        break;
    case TcpSocketState::CA_LAST_STATE:
        stateName = "CA_LAST_STATE";
        break;
    default:
        stateName = "UNKNOWN";
        break;
    }
    return stateName;
}

std::string
TcpGymEnv::GetTcpCAEventName(const TcpSocketState::TcpCAEvent_t event)
{
    std::string eventName = "UNKNOWN";
    switch (event)
    {
    case TcpSocketState::CA_EVENT_TX_START:
        eventName = "CA_EVENT_TX_START";
        break;
    case TcpSocketState::CA_EVENT_CWND_RESTART:
        eventName = "CA_EVENT_CWND_RESTART";
        break;
    case TcpSocketState::CA_EVENT_COMPLETE_CWR:
        eventName = "CA_EVENT_COMPLETE_CWR";
        break;
    case TcpSocketState::CA_EVENT_LOSS:
        eventName = "CA_EVENT_LOSS";
        break;
    case TcpSocketState::CA_EVENT_ECN_NO_CE:
        eventName = "CA_EVENT_ECN_NO_CE";
        break;
    case TcpSocketState::CA_EVENT_ECN_IS_CE:
        eventName = "CA_EVENT_ECN_IS_CE";
        break;
    case TcpSocketState::CA_EVENT_DELAYED_ACK:
        eventName = "CA_EVENT_DELAYED_ACK";
        break;
    case TcpSocketState::CA_EVENT_NON_DELAYED_ACK:
        eventName = "CA_EVENT_NON_DELAYED_ACK";
        break;
    default:
        eventName = "UNKNOWN";
        break;
    }
    return eventName;
}

/*
Define action space
*/
Ptr<OpenGymSpace>
TcpGymEnv::GetActionSpace()
{
    uint32_t nActions = 6;

    Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace>(nActions);

    return space;
}

/*
Define game over condition
*/
bool
TcpGymEnv::GetGameOver()
{
    m_isGameOver = false;
    bool test = false;
    static float stepCounter = 0.0;
    stepCounter += 1;
    if (stepCounter == 10 && test)
    {
        m_isGameOver = true;
    }
    NS_LOG_INFO("MyGetGameOver: " << m_isGameOver);
    return m_isGameOver;
}

/*
Define reward function
*/
float
TcpGymEnv::GetReward()
{
    NS_LOG_INFO("MyGetReward: " << m_envReward);
    return m_envReward;
}

/*
Define extra info. Optional
*/
std::string
TcpGymEnv::GetExtraInfo()
{
    NS_LOG_INFO("MyGetExtraInfo: " << m_info);
    return m_info;
}

/*
Execute received actions
*/
bool
TcpGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> act = DynamicCast<OpenGymDiscreteContainer>(action);

    m_action = act->GetValue();

    // NS_LOG_UNCOND("Selected Action = " << m_action);
    NS_LOG_INFO("MyExecuteActions: " << action);
    return true;
}

NS_OBJECT_ENSURE_REGISTERED(TcpEventGymEnv);

TcpEventGymEnv::TcpEventGymEnv()
    : TcpGymEnv()
{
    NS_LOG_FUNCTION(this);
}

TcpEventGymEnv::~TcpEventGymEnv()
{
    NS_LOG_FUNCTION(this);
}

TypeId
TcpEventGymEnv::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::TcpEventGymEnv")
                            .SetParent<TcpGymEnv>()
                            .SetGroupName("OpenGym")
                            .AddConstructor<TcpEventGymEnv>();

    return tid;
}

void
TcpEventGymEnv::DoDispose()
{
    NS_LOG_FUNCTION(this);
}

void
TcpEventGymEnv::SetReward(float value)
{
    NS_LOG_FUNCTION(this);
    m_reward = value;
}

void
TcpEventGymEnv::SetPenalty(float value)
{
    NS_LOG_FUNCTION(this);
    m_penalty = value;
}

/*
Define observation space
*/
Ptr<OpenGymSpace>
TcpEventGymEnv::GetObservationSpace()
{
    // socket unique ID
    // tcp env type: event-based = 0 / time-based = 1
    // sim time in us
    // node ID
    // ssThresh
    // cWnd
    // segmentSize
    // segmentsAcked
    // bytesInFlight
    // rtt in us
    // min rtt in us
    // called func
    // congetsion algorithm (CA) state
    // CA event
    // ECN state
    uint32_t parameterNum = 15;
    float low = 0.0;
    float high = 1000000000.0;
    std::vector<uint32_t> shape = {
        parameterNum,
    };
    std::string dtype = TypeNameGet<uint64_t>();

    Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    NS_LOG_INFO("MyGetObservationSpace: " << box);
    return box;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer>
TcpEventGymEnv::GetObservation()
{
    // NS_LOG_UNCOND("tcb->m_bytesInFlight = " << m_tcb->m_bytesInFlight);
// NS_LOG_UNCOND("stored m_bytesInFlight = " << m_bytesInFlight);
    uint32_t parameterNum = 15;
    std::vector<uint32_t> shape = {
        parameterNum,
    };


    Ptr<OpenGymBoxContainer<uint64_t>> box = CreateObject<OpenGymBoxContainer<uint64_t>>(shape);
    box->AddValue(m_socketUuid); //0
    box->AddValue(0);//1
    box->AddValue(Simulator::Now().GetMicroSeconds());//2->
    box->AddValue(m_nodeId);//3
    box->AddValue(m_tcb->m_ssThresh);//4
    box->AddValue(m_tcb->m_cWnd);//5->
    box->AddValue(m_tcb->m_segmentSize);//6->
    box->AddValue(m_segmentsAcked);//7->
    box->AddValue(m_tcb->m_bytesInFlight);//8->
    box->AddValue(m_rtt.GetMicroSeconds());//9->
    box->AddValue(m_tcb->m_minRtt.GetMicroSeconds());//10->
    box->AddValue(m_calledFunc);//11
    box->AddValue(m_tcb->m_congState);//12
    box->AddValue(m_event);//13
    box->AddValue(m_tcb->m_ecnState);//14

    // Print data
    // NS_LOG_UNCOND("MyGetObservation: " << box);
    return box;
}

void
TcpEventGymEnv::TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this);
}

void
TcpEventGymEnv::RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this);
}

uint32_t
TcpEventGymEnv::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    m_envReward = m_penalty;
    // NS_LOG_UNCOND("GetSsThresh called");
    m_calledFunc = CalledFunc_t::GET_SS_THRESH;
    m_info = "GetSsThresh";
    m_tcb = tcb;
    m_bytesInFlight = bytesInFlight;

    Notify();

    uint32_t MSS = tcb->m_segmentSize;
    uint32_t cwnd = tcb->m_cWnd;
    uint32_t newThresh = std::max(2 * MSS, cwnd / 2);

    return newThresh;
}

void
TcpEventGymEnv::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{
    // NS_LOG_UNCOND("m_bytesInFlight = " << tcb->m_bytesInFlight);
    // NS_LOG_UNCOND("BytesInFlight arg = " << m_bytesInFlight);
    // NS_LOG_UNCOND("EVENT ENV IncreaseWindow");
    NS_LOG_FUNCTION(this);
    //     NS_LOG_UNCOND(
    //     "Action=" << m_action
    //     << " CWND_before=" << tcb->m_cWnd
    // );
    // pkt was acked, so reward
    m_envReward = m_reward;
    uint32_t cwnd = tcb->m_cWnd;
    uint32_t MSS = tcb->m_segmentSize;
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId
                                 << " IncreaseWindow, SegmentsAcked: " << segmentsAcked);
    m_calledFunc = CalledFunc_t::INCREASE_WINDOW;
    m_info = "IncreaseWindow";
    m_tcb = tcb;
    m_segmentsAcked = segmentsAcked;
    // NS_LOG_UNCOND("CWND          = " << m_tcb->m_cWnd);
    // NS_LOG_UNCOND("BytesInFlight = " << m_bytesInFlight);
    // NS_LOG_UNCOND("MSS           = " << m_tcb->m_segmentSize);
    // NS_LOG_UNCOND("ssThresh = " << tcb->m_ssThresh);
    Notify();
    switch (m_action)
    {
    case 0: // exponential increase by MSS
        cwnd += 2 * MSS;
        break;

    case 1: // linear increase by MSS
        cwnd += MSS;
        break;

    case 2: // maintain
        break;

    case 3: // linear decrease by MSS
        if (cwnd > MSS)
            cwnd -= MSS;
        else
            cwnd = MSS;
        break;

    case 4: // exponential decrease by MSS
        if (cwnd > 2 * MSS)
            cwnd -= 2 * MSS;
        else
            cwnd = MSS;
        break;

    case 5: // recover to ssthresh
        cwnd = std::max(MSS, cwnd / 2);
        break;
    }

    tcb->m_cWnd = cwnd;
    //     NS_LOG_UNCOND(
    //     "CWND_after=" << cwnd
    // );
}

void
TcpEventGymEnv::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " PktsAcked, SegmentsAcked: "
                                 << segmentsAcked << " Rtt: " << rtt);
    m_calledFunc = CalledFunc_t::PKTS_ACKED;
    m_info = "PktsAcked";
    m_tcb = tcb;
    m_segmentsAcked = segmentsAcked;
    m_rtt = rtt;
}

void
TcpEventGymEnv::CongestionStateSet(Ptr<TcpSocketState> tcb,
                                   const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this);
    std::string stateName = GetTcpCongStateName(newState);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CongestionStateSet: " << newState
                                 << " " << stateName);

    m_calledFunc = CalledFunc_t::CONGESTION_STATE_SET;
    m_info = "CongestionStateSet";
    m_tcb = tcb;
    m_newState = newState;
}

void
TcpEventGymEnv::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this);
    std::string eventName = GetTcpCAEventName(event);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CwndEvent: " << event << " "
                                 << eventName);
    if(event == TcpSocketState::CA_EVENT_LOSS){
    m_lostEvent++;}
    m_calledFunc = CalledFunc_t::CWND_EVENT;
    m_info = "CwndEvent";
    m_tcb = tcb;
    m_event = event;
}

NS_OBJECT_ENSURE_REGISTERED(TcpTimeStepGymEnv);

TcpTimeStepGymEnv::TcpTimeStepGymEnv()
    : TcpGymEnv()
{
    NS_LOG_FUNCTION(this);
    m_envReward = 0.0;
}

TcpTimeStepGymEnv::TcpTimeStepGymEnv(Time timeStep)
    : TcpGymEnv()
{
    NS_LOG_FUNCTION(this);
    m_timeStep = timeStep;
    m_envReward = 0.0;
}

void
TcpTimeStepGymEnv::ScheduleNextStateRead()
{
    NS_LOG_FUNCTION(this);
    Simulator::Schedule(m_timeStep, &TcpTimeStepGymEnv::ScheduleNextStateRead, this);
    Notify();
}

TcpTimeStepGymEnv::~TcpTimeStepGymEnv()
{
    NS_LOG_FUNCTION(this);
}

TypeId
TcpTimeStepGymEnv::GetTypeId(void)
{
    static TypeId tid = TypeId("ns3::TcpTimeStepGymEnv")
                            .SetParent<TcpGymEnv>()
                            .SetGroupName("OpenGym")
                            .AddConstructor<TcpTimeStepGymEnv>();

    return tid;
}

void
TcpTimeStepGymEnv::DoDispose()
{
    NS_LOG_FUNCTION(this);
}

/*
Define observation space
*/
Ptr<OpenGymSpace>
TcpTimeStepGymEnv::GetObservationSpace()
{
    // socket unique ID
    // tcp env type: event-based = 0 / time-based = 1
    // sim time in us
    // node ID
    // ssThresh
    // cWnd
    // segmentSize
    // bytesInFlightSum
    // bytesInFlightAvg
    // segmentsAckedSum
    // segmentsAckedAvg
    // avgRtt
    // minRtt
    // avgInterTx
    // avgInterRx
    // throughput
    uint32_t parameterNum = 16;
    float low = 0.0;
    float high = 1000000000.0;
    std::vector<uint32_t> shape = {
        parameterNum,
    };
    std::string dtype = TypeNameGet<uint64_t>();

    Ptr<OpenGymBoxSpace> box = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    NS_LOG_INFO("MyGetObservationSpace: " << box);
    return box;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer>
TcpTimeStepGymEnv::GetObservation()
{
    uint32_t parameterNum = 16;
    std::vector<uint32_t> shape = {
        parameterNum,
    };

    Ptr<OpenGymBoxContainer<uint64_t>> box = CreateObject<OpenGymBoxContainer<uint64_t>>(shape);

    box->AddValue(m_socketUuid);
    box->AddValue(1);
    box->AddValue(Simulator::Now().GetMicroSeconds());
    box->AddValue(m_nodeId);
    box->AddValue(m_tcb->m_ssThresh);
    box->AddValue(m_tcb->m_cWnd);
    box->AddValue(m_tcb->m_segmentSize);

    // bytesInFlightSum
    uint64_t bytesInFlightSum = std::accumulate(m_bytesInFlight.begin(), m_bytesInFlight.end(), 0);
    box->AddValue(bytesInFlightSum);

    // bytesInFlightAvg
    uint64_t bytesInFlightAvg = 0;
    if (m_bytesInFlight.size())
    {
        bytesInFlightAvg = bytesInFlightSum / m_bytesInFlight.size();
    }
    box->AddValue(bytesInFlightAvg);

    // segmentsAckedSum
    uint64_t segmentsAckedSum = std::accumulate(m_segmentsAcked.begin(), m_segmentsAcked.end(), 0);
    box->AddValue(segmentsAckedSum);

    // segmentsAckedAvg
    uint64_t segmentsAckedAvg = 0;
    if (m_segmentsAcked.size())
    {
        segmentsAckedAvg = segmentsAckedSum / m_segmentsAcked.size();
    }
    box->AddValue(segmentsAckedAvg);

    // avgRtt
    Time avgRtt = Seconds(0.0);
    if (m_rttSampleNum)
    {
        avgRtt = m_rttSum / m_rttSampleNum;
    }
    box->AddValue(avgRtt.GetMicroSeconds());

    // m_minRtt
    box->AddValue(m_tcb->m_minRtt.GetMicroSeconds());

    // avgInterTx
    Time avgInterTx = Seconds(0.0);
    if (m_interTxTimeNum)
    {
        avgInterTx = m_interTxTimeSum / m_interTxTimeNum;
    }
    box->AddValue(avgInterTx.GetMicroSeconds());

    // avgInterRx
    Time avgInterRx = Seconds(0.0);
    if (m_interRxTimeNum)
    {
        avgInterRx = m_interRxTimeSum / m_interRxTimeNum;
    }
    box->AddValue(avgInterRx.GetMicroSeconds());

    // throughput  bytes/s
    float throughput = (segmentsAckedSum * m_tcb->m_segmentSize) / m_timeStep.GetSeconds();
    box->AddValue(throughput);

    // Print data
    NS_LOG_INFO("MyGetObservation: " << box);

    m_bytesInFlight.clear();
    m_segmentsAcked.clear();

    m_rttSampleNum = 0;
    m_rttSum = MicroSeconds(0.0);

    m_interTxTimeNum = 0;
    m_interTxTimeSum = MicroSeconds(0.0);

    m_interRxTimeNum = 0;
    m_interRxTimeSum = MicroSeconds(0.0);

    return box;
}

void
TcpTimeStepGymEnv::TxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this);
    if (m_lastPktTxTime > MicroSeconds(0.0))
    {
        Time interTxTime = Simulator::Now() - m_lastPktTxTime;
        m_interTxTimeSum += interTxTime;
        m_interTxTimeNum++;
    }

    m_lastPktTxTime = Simulator::Now();
}

void
TcpTimeStepGymEnv::RxPktTrace(Ptr<const Packet>, const TcpHeader&, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this);
    if (m_lastPktRxTime > MicroSeconds(0.0))
    {
        Time interRxTime = Simulator::Now() - m_lastPktRxTime;
        m_interRxTimeSum += interRxTime;
        m_interRxTimeNum++;
    }

    m_lastPktRxTime = Simulator::Now();
}

uint32_t
TcpTimeStepGymEnv::GetSsThresh(Ptr<const TcpSocketState> tcb, uint32_t bytesInFlight)
{
    // NS_LOG_UNCOND("GetSsThresh called");
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId
                                 << " GetSsThresh, BytesInFlight: " << bytesInFlight);
    m_tcb = tcb;
    m_bytesInFlight.push_back(bytesInFlight);

    if (!m_started)
    {
        m_started = true;
        Notify();
        ScheduleNextStateRead();
    }

    // action
    return m_new_ssThresh;
}

void
TcpTimeStepGymEnv::IncreaseWindow(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked)
{

    NS_LOG_UNCOND("TIMESTEP ENV IncreaseWindow");
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId
                                 << " IncreaseWindow, SegmentsAcked: " << segmentsAcked);
    m_tcb = tcb;
    m_segmentsAcked.push_back(segmentsAcked);
    m_bytesInFlight.push_back(tcb->m_bytesInFlight);

    if (!m_started)
    {
        m_started = true;
        Notify();
        ScheduleNextStateRead();
    }
    // action
    tcb->m_cWnd = m_new_cWnd;
}

void
TcpTimeStepGymEnv::PktsAcked(Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt)
{
    NS_LOG_FUNCTION(this);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " PktsAcked, SegmentsAcked: "
                                 << segmentsAcked << " Rtt: " << rtt);
    m_tcb = tcb;
    m_rttSum += rtt;
    m_rttSampleNum++;
}

void
TcpTimeStepGymEnv::CongestionStateSet(Ptr<TcpSocketState> tcb,
                                      const TcpSocketState::TcpCongState_t newState)
{
    NS_LOG_FUNCTION(this);
    std::string stateName = GetTcpCongStateName(newState);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CongestionStateSet: " << newState
                                 << " " << stateName);
    m_tcb = tcb;
}

void
TcpTimeStepGymEnv::CwndEvent(Ptr<TcpSocketState> tcb, const TcpSocketState::TcpCAEvent_t event)
{
    NS_LOG_FUNCTION(this);
    std::string eventName = GetTcpCAEventName(event);
    NS_LOG_INFO(Simulator::Now() << " Node: " << m_nodeId << " CwndEvent: " << event << " "
                                 << eventName);
    m_tcb = tcb;
}

} // namespace ns3
